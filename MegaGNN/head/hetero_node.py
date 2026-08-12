import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from MegaGNN.graphgym.register import register_head
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.models.layer import MLP
from torch_geometric.utils import mask_to_index
from torch_geometric.nn import Linear


@register_head('hetero_node')
class HeteroGNNNodeHead(nn.Module):
    r'''Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    '''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        self.layer_post_mp = nn.Sequential(Linear(dim_in, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
                                Linear(25, dim_out))

    def _apply_index(self, batch):
        
        task = cfg.dataset.task_entity
        if batch.split == 'val':
            input_id = batch[task].input_id - (self.train_inds.max() + 1)
        elif batch.split == 'test':
            input_id = batch[task].input_id - (self.val_inds.max() + 1)
        else:
            input_id = batch[task].input_id
        
        mask = torch.isin(batch[task].n_id,
                          getattr(self, f'{batch.split}_inds')[input_id])
        
        return batch[task].x[mask], batch[task].y[mask]

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label

@register_head('hetero_node_synthetic')
class HeteroGNNNodeHeadSynthetic(nn.Module):
    r'''Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    '''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        self.layer_post_mp = nn.Sequential(Linear(dim_in, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
                                Linear(25, 11))

    def _apply_index(self, batch):
        
        task = cfg.dataset.task_entity
        input_id = batch[task].input_id
        
        mask = torch.isin(batch[task].n_id,
                          getattr(self, f'{batch.split}_inds')[input_id])
        
        return batch[task].x[mask], batch[task].y[mask]

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label

    

@register_head('hetero_node_per_neighbor')
class HeteroGNNNodeHeadPerNeighbor(nn.Module):
    r'''Head for per-split-graph node prediction (no input_id offset).
    Works for any dataset where each split is an independent graph.
    NeighborLoader places seed nodes first, so the first batch_size
    entries are always the target nodes.
    '''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()

        self.layer_post_mp = nn.Sequential(
            Linear(dim_in, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
            Linear(25, dim_out),
        )

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        bs = batch[task].batch_size
        return batch[task].x[:bs], batch[task].y[:bs]

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label