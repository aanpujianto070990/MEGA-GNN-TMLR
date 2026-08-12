import logging
import os, time
import os.path as osp
import torch

from MegaGNN.graphgym.register import register_loader

from MegaGNN.datasets.aml_dataset import AMLDataset
from MegaGNN.datasets.eth_kaggle_dataset import ETHKaggleDataset
from MegaGNN.datasets.jodie_dataset import JodieDataset
from MegaGNN.datasets.synthetic_pattern import SyntheticPatternDataset
from MegaGNN.datasets.synthetic_per_neighbor import SyntheticPerNeighborDataset

from MegaGNN.graphgym.config import cfg


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'num_nodes_dict'):
        total_num_nodes = sum(dataset.data.num_nodes_dict.values())
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")



@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    
    if format == 'AML':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_AML(dataset_dir, name)

    elif format == 'Jodie':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_Jodie(dataset_dir, name)

    elif format == 'ETH':
        dataset_dir = osp.join(dataset_dir, format)
        if name =='Kaggle':
            dataset = preformat_ETH_Kaggle(dataset_dir, name)
        else:
            raise ValueError(f"Unknown data format: {format}")

    elif format == 'synthetic_pattern':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_synthetic_pattern(dataset_dir, name)

    elif format == 'synthetic_per_neighbor':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_synthetic_per_neighbor(dataset_dir, name)

    else:
        raise ValueError(f"Unknown data format: {format}")

    # pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    return dataset


def preformat_synthetic_pattern(dataset_dir, name):
    """Load and preformat custom Synthetic Pattern datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific Synthetic Pattern dataset
    Returns:
        SyntheticPatternDataset object
    """
    dataset = SyntheticPatternDataset(root=dataset_dir, name=name, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports, multi_edge_agg = cfg.gnn.multi_edge_agg)
    return dataset

def preformat_AML(dataset_dir, name):
    """Load and preformat custom Anti-money Laundering datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific AML dataset

    Returns:
        PyG dataset object
    """
    dataset = AMLDataset(root=dataset_dir, name=name, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports, multi_edge_agg = cfg.gnn.multi_edge_agg)
    return dataset


def preformat_Jodie(dataset_dir, name):
    """Load and preformat custom Jodie datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific Jodie dataset
    
    Returns:
        PyG dataset object
    """
    dataset = JodieDataset(root=dataset_dir, name=name, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports, multi_edge_agg = cfg.gnn.multi_edge_agg)
    return dataset


def preformat_ETH_Kaggle(dataset_dir, name):
    """Load and preformat custom Ethereum Phishing Detection datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific AML dataset

    Returns:
        PyG dataset object
    """
    dataset = ETHKaggleDataset(root=dataset_dir, name=name, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports, multi_edge_agg = cfg.gnn.multi_edge_agg, collapse_rate=cfg.dataset.collapse_rate)
    return dataset


def preformat_synthetic_per_neighbor(dataset_dir, name):
    """Load and preformat synthetic per-neighbor statistics benchmark.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific task variant (e.g. dominant_sender_sum)

    Returns:
        SyntheticPerNeighborDataset object
    """
    dataset = SyntheticPerNeighborDataset(
        root=dataset_dir, name=name,
        reverse_mp=cfg.dataset.reverse_mp,
        add_ports=cfg.dataset.add_ports,
        multi_edge_agg=cfg.gnn.multi_edge_agg,
    )
    return dataset