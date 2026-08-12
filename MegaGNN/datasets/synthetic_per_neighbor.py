"""
Synthetic per-neighbor statistics benchmark dataset.

This dataset now exposes two label families generated from the same directed
multigraph:

  - Family A: current sender-total targets plus bundle-max targets
  - Family B: sender-neighborhood targets derived from each sender's own
    incoming neighborhood total before regrouping at the receiver

Single-stage GNNs see incoming edges as a flat bag. Multi-edge and iterative
grouping models can exploit the structured sender-wise grouping signal.
"""

import logging
import os.path as osp
import random
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from .synthetic_pattern import find_parallel_edges
from .temporal_dataset import TemporalDataset

from MegaGNN.graphgym.config import cfg


DATASET_PARAMS = {
    'per_neighbor_stats': {
        'num_nodes': 8192,
        'm_ba': 3,
        'max_parallel': 5,
        'lognormal_mu': 1.0,
        'lognormal_sigma': 0.8,
        'min_in_neighbors': 2,
    },
    'per_neighbor_stats_large': {
        'num_nodes': 10000,
        'm_ba': 4,
        'max_parallel': 6,
        'lognormal_mu': 1.0,
        'lognormal_sigma': 1.0,
        'min_in_neighbors': 2,
    },
}

TARGET_NAMES = {
    'A': [
        'max_sender_sum',
        'sender_sum_var',
        'top_gap',
        'sum_sender_maxima',
        'std_sender_maxima',
    ],
    'B': [
        'max_sender_neighbor_sum',
        'sender_neighbor_sum_var',
        'sender_neighbor_top_gap',
    ],
}


def _compute_sender_group_stats(senders: np.ndarray,
                                vals: np.ndarray,
                                unique_senders: np.ndarray) -> Dict[str, np.ndarray]:
    sender_sums = np.array([vals[senders == s].sum() for s in unique_senders])
    sender_maxima = np.array([vals[senders == s].max() for s in unique_senders])
    return {
        'sender_sums': sender_sums,
        'sender_maxima': sender_maxima,
    }


def _fill_top_stats(target: torch.Tensor, values: np.ndarray) -> None:
    sorted_values = np.sort(values)[::-1]
    target[0] = float(sorted_values[0])
    target[1] = float(np.var(values))
    target[2] = float(sorted_values[0] - sorted_values[1])


def generate_per_neighbor_graph(
    num_nodes: int,
    m_ba: int,
    max_parallel: int,
    lognormal_mu: float,
    lognormal_sigma: float,
    min_in_neighbors: int,
    seed: int = 0,
) -> Dict[str, object]:
    """Generate one split of the dual-family synthetic benchmark."""
    rng = np.random.RandomState(seed)
    random.seed(seed)

    nx_graph = nx.barabasi_albert_graph(num_nodes, m_ba, seed=seed)
    nx_digraph = nx.DiGraph([random.sample(list(e), 2) for e in nx_graph.edges()])
    base_edges = np.array(nx_digraph.edges())

    multiplicities = rng.randint(1, max_parallel + 1, size=len(base_edges))
    src = np.repeat(base_edges[:, 0], multiplicities)
    dst = np.repeat(base_edges[:, 1], multiplicities)
    amounts = rng.lognormal(lognormal_mu, lognormal_sigma, size=len(src))

    # Sender-neighborhood score for Family B: each node's total incoming amount.
    node_incoming_totals = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(node_incoming_totals, dst, amounts)

    label_bank = {
        'A': torch.zeros(num_nodes, len(TARGET_NAMES['A']), dtype=torch.float),
        'B': torch.zeros(num_nodes, len(TARGET_NAMES['B']), dtype=torch.float),
    }
    mask_bank = {
        'A': torch.zeros(num_nodes, dtype=torch.bool),
        'B': torch.zeros(num_nodes, dtype=torch.bool),
    }

    for node in range(num_nodes):
        incoming = np.where(dst == node)[0]
        if len(incoming) == 0:
            continue

        senders = src[incoming]
        vals = amounts[incoming]
        unique_senders = np.unique(senders)

        if len(unique_senders) < min_in_neighbors:
            continue

        group_stats = _compute_sender_group_stats(senders, vals, unique_senders)
        sender_sums = group_stats['sender_sums']
        sender_maxima = group_stats['sender_maxima']
        sender_neighbor_scores = node_incoming_totals[unique_senders]

        _fill_top_stats(label_bank['A'][node, :3], sender_sums)
        label_bank['A'][node, 3] = float(sender_maxima.sum())
        label_bank['A'][node, 4] = float(np.std(sender_maxima))
        mask_bank['A'][node] = True

        _fill_top_stats(label_bank['B'][node], sender_neighbor_scores)
        mask_bank['B'][node] = True

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(amounts, dtype=torch.float).unsqueeze(1)

    logging.info(
        f"  BA graph: {num_nodes} nodes, {len(base_edges)} base edges, "
        f"{len(src)} total edges (after parallel), "
        f"A-targets={mask_bank['A'].sum().item()}, B-targets={mask_bank['B'].sum().item()}"
    )

    for family in ['A', 'B']:
        sample_ids = torch.where(mask_bank[family])[0][:3].tolist()
        for sid in sample_ids:
            logging.info(
                f"    {family} node {sid}: {label_bank[family][sid].tolist()}"
            )

    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes,
        'label_bank': label_bank,
        'mask_bank': mask_bank,
    }


class SyntheticPerNeighborDataset(TemporalDataset):
    """Synthetic per-neighbor statistics benchmark on BA multigraphs.

    Each split is an independent directed multigraph generated from a
    Barabasi-Albert model.  Node features are constant; all signal is in
    edge amounts and their grouping by sender.
    """

    def __init__(
        self,
        root: str,
        name: str,
        reverse_mp: bool = False,
        add_ports: bool = False,
        multi_edge_agg: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        self.multi_edge_agg = multi_edge_agg
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0], weights_only=False)
        self.norm_stats = torch.load(self.processed_paths[2], weights_only=False)

        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                if ('node', 'rev_to', 'node') in self.data_dict[split].edge_types:
                    del self.data_dict[split]['node', 'rev_to', 'node']

        if multi_edge_agg:
            self.simp_edge_batch_dict = torch.load(
                self.processed_paths[1], weights_only=False
            )
            for split in ['train', 'val', 'test']:
                for edge_type in self.data_dict[split].edge_types:
                    self.data_dict[split][edge_type].simp_edge_batch = \
                        self.simp_edge_batch_dict[split]

        self._apply_label_family()

    def _apply_label_family(self) -> None:
        family = str(cfg.dataset.synthetic_label_family).upper()
        if family not in TARGET_NAMES:
            raise ValueError(
                f"Unknown synthetic label family '{cfg.dataset.synthetic_label_family}'. "
                f"Expected one of {list(TARGET_NAMES.keys())}"
            )

        for split in ['train', 'val', 'test']:
            node_store = self.data_dict[split]['node']
            node_store.y = getattr(node_store, f'y_{family}')
            node_store.split_mask = getattr(node_store, f'split_mask_{family}')
            node_store.target_names = TARGET_NAMES[family]
            node_store.active_label_family = family

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'simp_edge_batch.pt', 'norm_stats.pt']

    def process(self):
        params = DATASET_PARAMS.get(
            self.name, DATASET_PARAMS['per_neighbor_stats']
        )

        split_seeds = {'train': 42, 'val': 43, 'test': 44}
        raw_graphs = {}

        # --- generate all splits ---
        for split, seed in split_seeds.items():
            logging.info(f"Generating {split} split (seed={seed})...")
            raw_graphs[split] = generate_per_neighbor_graph(
                num_nodes=params['num_nodes'],
                m_ba=params['m_ba'],
                max_parallel=params['max_parallel'],
                lognormal_mu=params['lognormal_mu'],
                lognormal_sigma=params['lognormal_sigma'],
                min_in_neighbors=params['min_in_neighbors'],
                seed=seed,
            )

        norm_stats = {}
        for family in TARGET_NAMES:
            train_mask = raw_graphs['train']['mask_bank'][family]
            train_y = raw_graphs['train']['label_bank'][family][train_mask]
            y_mean = train_y.mean(dim=0)
            y_std = train_y.std(dim=0).clamp(min=1e-6)
            norm_stats[family] = {
                'mean': y_mean,
                'std': y_std,
                'target_names': TARGET_NAMES[family],
            }
            logging.info(
                f"  {family} normalization: mean={y_mean.tolist()}, std={y_std.tolist()}"
            )

            for split in ['train', 'val', 'test']:
                mask = raw_graphs[split]['mask_bank'][family]
                raw_graphs[split]['label_bank'][family][mask] = (
                    (raw_graphs[split]['label_bank'][family][mask] - y_mean) / y_std
                )

        # --- convert to HeteroData and compute simp_edge_batch ---
        self.data_dict = {}
        self.simp_edge_batch_dict = {}

        for split in ['train', 'val', 'test']:
            raw = raw_graphs[split]
            N = raw['num_nodes']

            data = HeteroData()
            data['node'].num_nodes = N
            data['node'].x = torch.ones((N, 1), dtype=torch.float)
            for family in TARGET_NAMES:
                data['node'][f'y_{family}'] = raw['label_bank'][family]
                data['node'][f'split_mask_{family}'] = raw['mask_bank'][family]

            data['node', 'to', 'node'].edge_index = raw['edge_index']
            data['node', 'to', 'node'].edge_attr = raw['edge_attr']

            data['node', 'rev_to', 'node'].edge_index = raw['edge_index'].flipud()
            data['node', 'rev_to', 'node'].edge_attr = raw['edge_attr'].clone()

            simp_edge_batch = find_parallel_edges(raw['edge_index'])

            self.data_dict[split] = data
            self.simp_edge_batch_dict[split] = simp_edge_batch

            logging.info(
                f"  {split}: {N} nodes, {raw['edge_index'].shape[1]} edges, "
                f"A-targets={raw['mask_bank']['A'].sum().item()}, "
                f"B-targets={raw['mask_bank']['B'].sum().item()}"
            )

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.simp_edge_batch_dict, self.processed_paths[1])
        torch.save(norm_stats, self.processed_paths[2])

    def __repr__(self) -> str:
        return f'SyntheticPerNeighborDataset(name={self.name})'
