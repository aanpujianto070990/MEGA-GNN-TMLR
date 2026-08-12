import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.register import register_node_encoder, register_edge_encoder



@register_node_encoder('Hetero_Raw')
class HeteroRawNodeEncoder(torch.nn.Module):
    """
    The raw feature node encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        if cfg.train.add_ego_id:
            if cfg.dataset.name == "pattern":
                self.dim_in = {
                    node_type: self.dim_in[node_type] + cfg.train.batch_size for node_type in self.dim_in
                }
            else:   
                self.dim_in = {
                    node_type: self.dim_in[node_type] + 1 for node_type in self.dim_in
                }
        self.metadata = dataset[0].metadata()
        data = dataset[0]
        
        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in data.node_types}
        
        self.linear = nn.ModuleDict()
        for node_type in self.metadata[0]:
            if hasattr(data[node_type], 'x'):
                self.linear[node_type] = nn.Linear(
                    self.dim_in[node_type], dim_emb[node_type])
            else:
                self.dim_in[node_type] = dim_emb[node_type]

        self.encoder = nn.ModuleDict(
            {
                node_type: nn.Embedding(data[node_type].num_nodes, dim_emb[node_type])
                for node_type in data.node_types
                if not hasattr(data[node_type], 'x')
            }
        )
        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'x'):
                    batch[node_type].x = self.linear[node_type](batch[node_type].x)
                else:
                    if cfg.train.sampler == 'full_batch':
                        batch[node_type].x = self.encoder[node_type].weight.data #(torch.arange(batch[node_type].num_nodes))
                    else:
                        batch[node_type].x = self.encoder[node_type](batch[node_type].n_id)
        else:
            x = batch.x
            batch.x = list(self.linear.values())[0](x)

        return batch


@register_edge_encoder('Hetero_Raw')
class HeteroRawEdgeEncoder(torch.nn.Module):
    """
    The raw feature edge encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        self.metadata = dataset[0].metadata()
        
        if not isinstance(dim_emb, dict):
            dim_emb = {edge_type: dim_emb for edge_type in self.dim_in}
        
        self.linear = nn.ModuleDict()
        for edge_type in self.metadata[1]:
            edge_type = '__'.join(edge_type)
            self.linear[edge_type] = nn.Linear(
                self.dim_in[edge_type], dim_emb[edge_type])
        
        if cfg.model.type == 'FusionModel':
            self.task = cfg.dataset.task_entity
            edge_type = '__'.join(self.task)

            if cfg.pearl.enable and cfg.pearl.type == 'concat':
                self.linear_attn = nn.Linear(self.dim_in[edge_type]+2*cfg.pearl.dim_pe, dim_emb[edge_type])
            elif cfg.pearl.enable and cfg.pearl.type == 'add':
                self.linear_attn = nn.Linear(self.dim_in[edge_type], dim_emb[edge_type])
                self.linear_pe = nn.Linear(2*cfg.pearl.dim_pe, dim_emb[edge_type])
            else:
                self.linear_attn = nn.Linear(self.dim_in[edge_type], dim_emb[edge_type])
    
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            if cfg.model.type == 'FusionModel':
                if cfg.pearl.enable:
                    src, dst = batch[self.task].edge_index
                    if cfg.pearl.type == 'concat':    
                        batch[self.task].edge_attr_attn = self.linear_attn(
                            torch.cat([batch[self.task].edge_attr, batch['node'].pearl_pe[src], batch['node'].pearl_pe[dst]], dim=1)
                            )
                    elif cfg.pearl.type == 'add':
                        batch[self.task].edge_attr_attn = self.linear_attn(batch[self.task].edge_attr)
                        pe = torch.cat([batch['node'].pearl_pe[src], batch['node'].pearl_pe[dst]], 1)
                        batch[self.task].edge_attr_attn = batch[self.task].edge_attr_attn + self.linear_pe(pe)
                else:
                    batch[self.task].edge_attr_attn = self.linear_attn(batch[self.task].edge_attr)
                
            for edge_type, edge_attr in batch.collect("edge_attr").items():
                batch[edge_type].edge_attr = self.linear['__'.join(edge_type)](edge_attr) 
        
        else:
            edge_attr = batch.edge_attr
            batch.edge_attr = list(self.linear.values())[0](edge_attr)

        return batch