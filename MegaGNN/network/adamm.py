import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder

from torch_geometric.data import HeteroData
from MegaGNN.graphgym import register as register
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.models.gnn import GNNPreMP
from MegaGNN.graphgym.register import register_network
from MegaGNN.graphgym.models.layer import BatchNorm1dNode

from MegaGNN.layer.rgcn_conv import FastRGCNConv
from MegaGNN.layer.multiedgeagg import MultiEdgeAggregation

from torch_geometric.nn import (HeteroConv, PNAConv, GINEConv)
from torch_geometric.utils import degree


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dataset):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, dataset)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension of the node features
            if self.is_hetero:
                self.dim_in = {node_type: cfg.gnn.dim_inner for node_type in dim_in}
            else:
                self.dim_in = cfg.gnn.dim_hidden
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gnn.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, dataset)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
    

def ToAdammRepresentation(data): 
    homo_data = data.to_homogeneous()
    edge_index, edge_attr = homo_data.edge_index, homo_data.edge_attr
    
    device = edge_index.device

    # create full edge_index, assume that original input edge_index is single direction directed, mult-edge is allowed
    self_loop_indice = edge_index[0] == edge_index[1]
    self_loops = edge_index[:, self_loop_indice]
    other_edges = edge_index[:, ~self_loop_indice]
    reversed_other_edges = torch.stack([other_edges[1], other_edges[0]])

    edge_index = torch.cat([self_loops, other_edges, reversed_other_edges], dim=-1).to(device)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr[self_loop_indice], edge_attr[~self_loop_indice], edge_attr[~self_loop_indice]], dim=0).to(device)
    edge_direction = torch.cat([torch.full((self_loops.size(-1),), 0), torch.full((other_edges.size(-1),), 1), torch.full((reversed_other_edges.size(-1),), 2)], dim=0).to(device)
    

    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        # transform edge to tuple
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i

            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch).to(device)

    homo_data.edge_index = edge_index
    homo_data.edge_attr = edge_attr
    homo_data.edge_direction = edge_direction
    homo_data.simp_edge_batch = simplified_edge_batch
    return homo_data


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out = out + self.embeddings[i](x[:, i])
        return out
    


@register_network('AdammModel')
class GNNEdgeModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata = dataset[0].metadata()

        # Initialize basic components
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.encoder = FeatureEncoder(dim_in, dataset)
        self.multi_edge_agg = cfg.gnn.multi_edge_agg
        self.layer_type = cfg.gnn.layer_type
        

        if cfg.gnn.multi_edge_agg_type =='adamm':
            self.edge_direction_encoder = DiscreteEncoder(cfg.gnn.dim_inner, max_num_values=4)
        else:
            raise RuntimeError("AdammModel only supports adamm as multi_edge_agg_type")

        dim_in = self.encoder.dim_in
        dim_h_total = cfg.gnn.dim_inner

        # Initialize pre-message passing layers if needed
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                cfg.gnn.dim_inner, cfg.gnn.dim_inner,
                has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
            )

        # Initialize GNN helper with proper heterogeneous handling
        self.gnn = GNNHelper(dataset)

        # Initialize post-message passing head
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h_total, dim_out, dataset)

    def forward(self, batch):
        # Encode features
        batch = self.encoder(batch)

        homo_batch = ToAdammRepresentation(batch)
        homo_batch.edge_attr = homo_batch.edge_attr + self.edge_direction_encoder(homo_batch.edge_direction)

        homo_batch.x = self.input_drop(homo_batch.x)

        homo_batch = self.gnn(homo_batch)


        batch['node'].x = homo_batch.x
        batch['node', 'to', 'node'].edge_attr = homo_batch.edge_attr


        return self.post_mp(batch)



class GNNHelper(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata = dataset[0].metadata()

        self.data = dataset[0]
        
        # Initialize basic components
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.activation = register.act_dict[cfg.gnn.act]
        
        # Initialize feature processing flags
        self.layer_norm = cfg.gnn.layer_norm
        self.batch_norm = cfg.gnn.batch_norm
        self.edge_updates = cfg.gnn.edge_updates
        self.multi_edge_agg = cfg.gnn.multi_edge_agg
        self.multi_edge_agg_type = cfg.gnn.multi_edge_agg_type
        self.layer_type = cfg.gnn.layer_type

        # Initialize network components
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        if self.layer_norm or self.batch_norm:
            self.norms = nn.ModuleList()

        for i in range(cfg.gnn.layers_mp):
            norm_dim = cfg.gnn.dim_inner
                
            self.convs.append(self._get_gnn_layer())

            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                            nn.Linear(3 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                            nn.ReLU(),
                            nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                        ))
            if self.layer_norm:
                self.norms.append(nn.LayerNorm(norm_dim))
            elif self.batch_norm:
                self.norms.append(nn.BatchNorm1d(norm_dim))


        self.edge_agg = MultiEdgeAggregation(
                        n_hidden=cfg.gnn.dim_inner,
                        agg_type=self.multi_edge_agg_type,
                        data=self.data
                    )

    def _get_gnn_layer(self):
        if cfg.gnn.layer_type == 'GINE':
            mlp = nn.Sequential(
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                    nn.ReLU(), 
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                )
            conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner)

        elif cfg.gnn.layer_type == 'GenAgg':
            mlp = nn.Sequential(
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                    nn.ReLU(), 
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                )
            conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner,
                            aggr=GenAgg(f=MLPAutoencoder, jit=False))
                            
        elif cfg.gnn.layer_type == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            if not isinstance(self.data, HeteroData):
                d = degree(self.data.edge_index[1], dtype=torch.long)
            else:
                d = degree(self.data.to_homogeneous().edge_index[1], dtype=torch.long)
            deg = torch.bincount(d, minlength=1)

            conv = PNAConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner,
                    aggregators=aggregators, scalers=scalers, deg=deg,
                    edge_dim=cfg.gnn.dim_inner, towers=1, pre_layers=1, post_layers=1,
                    divide_input=False)

                
        else:
            raise NotImplementedError(f"{cfg.gnn.layer_type} is not implemented!")
        
        return conv

    def forward(self, batch):

        x = batch.x
        edge_index, edge_attr, simp_edge_batch = batch.edge_index, batch.edge_attr, batch.simp_edge_batch

        # Apply the flattening at the beggining only.
        edge_index, edge_attr, _, _ = self.edge_agg(edge_index, edge_attr, simp_edge_batch)
        src, dst = edge_index

        for i in range(cfg.gnn.layers_mp):

            x_new = self.convs[i](x, edge_index, edge_attr)
            x_new = self.norms[i](x_new)
            x_new = self.activation(x_new)
            x_new = self.drop(x_new)

            if cfg.gnn.residual:
                x = (x + x_new) / 2
            else:
                x = x_new
            
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2   
        
        batch.x = x
        batch.edge_attr = edge_attr

        return batch
