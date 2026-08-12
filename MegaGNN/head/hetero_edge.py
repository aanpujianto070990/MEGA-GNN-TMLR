import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index
from torch_geometric.nn import Linear

from MegaGNN.graphgym.register import register_head
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.models.layer import MLP


@register_head('hetero_edge')
class HeteroGNNEdgeHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        # self.layer_post_mp = MLP(dim_in * 3, dim_out, 
        #                          num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
        #                          bias=True, dim_inner=dim_in)
        
        self.layer_post_mp = nn.Sequential(Linear(dim_in*3, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
                                Linear(25, dim_out))


    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        mask = torch.isin(batch[task].e_id, 
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        return torch.cat((batch[task[0]].x[edge_index[0, mask]], 
                          batch[task[2]].x[edge_index[1, mask]], 
                          batch[task].edge_attr[mask]), dim=-1), \
               batch[task].y[mask]
    

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label

@register_head('hetero_edge_fraudgt')
class HeteroGNNEdgeHeadFraudGT(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        self.layer_post_mp = MLP(dim_in * 3, dim_out, 
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True)


    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        mask = torch.isin(batch[task].e_id, 
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        return torch.cat((batch[task[0]].x[edge_index[0, mask]], 
                          batch[task[2]].x[edge_index[1, mask]], 
                          batch[task].edge_attr[mask]), dim=-1), \
               batch[task].y[mask]
    

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label


@register_head('hetero_edge_missing_rev')
class HeteroGNNEdgeMissingRevHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        # self.layer_post_mp = nn.Sequential(
        #     nn.Linear(dim_in*4, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
        #     nn.Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
        #     nn.Linear(25, dim_out)
        # )
        self.layer_post_mp = MLP(dim_in * 4, dim_out, 
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True, dim_inner=dim_in)


    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        reverse_task = (task[0], f'rev_{task[1]}', task[2])

        mask = torch.isin(batch[task].e_id, 
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        rev_mask = batch[reverse_task].rev_mask

        m_edge_index = batch[task].edge_index[:, mask]
        m_edge_attr = batch[task].edge_attr[mask, :]

        mrev_edge_index = batch[reverse_task].edge_index[:, rev_mask]
        mrev_edge_attr = batch[reverse_task].edge_attr[rev_mask, :]
        
        s_idx = torch.argsort(batch[task].e_id[mask])
        rs_idx = torch.argsort(batch[reverse_task].e_id[rev_mask])

        m_edge_index = m_edge_index[:, s_idx]
        mrev_edge_index = mrev_edge_index[:, rs_idx]
        m_edge_attr = m_edge_attr[s_idx]
        mrev_edge_attr = mrev_edge_attr[rs_idx]

        assert (m_edge_index[1] == mrev_edge_index[0]).all() and (m_edge_index[0] == mrev_edge_index[1]).all(), \
            "Forward and Reverse Edge indices are not aligned."

        x = torch.cat((batch[task[0]].x[m_edge_index[0, :]], 
                       batch[task[2]].x[m_edge_index[1, :]], 
                       m_edge_attr,
                       mrev_edge_attr), dim=-1)

        # A concatentation of source/target node embedding + edge attribute
        return x, batch[task].y[mask]
    

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label



@register_head('hetero_edge_full_batch')
class HeteroGNNEdgeHeadFullBatch(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        # self.layer_post_mp = MLP(dim_in * 3, dim_out, 
        #                          num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
        #                          bias=True, dim_inner=dim_in)
        
        self.layer_post_mp = nn.Sequential(Linear(dim_in*3, 50), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.gnn.dropout),
                                Linear(25, dim_out))


    def _apply_index(self, batch):
        task = cfg.dataset.task_entity

        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        return torch.cat((batch[task[0]].x[edge_index[0, :]], 
                          batch[task[2]].x[edge_index[1, :]], 
                          batch[task].edge_attr), dim=-1), \
               batch[task].y
    

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)
        return pred, label