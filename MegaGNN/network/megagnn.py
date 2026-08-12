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
from MegaGNN.layer.hetero_multiedgeagg import HeteroMultiEdgeAggregator

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

    
@register_network('MegaGNNModel')
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

        x_dict = batch.collect('x')

        x_dict = {
                node_type: self.input_drop(x) for node_type, x in x_dict.items()
            }

        if cfg.gnn.layers_pre_mp > 0:
            x_dict = {
                node_type: self.pre_mp_dict[node_type](x) for node_type, x in x_dict.items()
            }
        
        for node_type, x in x_dict.items():
            batch[node_type].x = x
        
        batch = self.gnn(batch)

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
        self.edge_aggrs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        if self.layer_norm or self.batch_norm:
            self.norms = nn.ModuleList()

        for i in range(cfg.gnn.layers_mp):
            norm_dim = cfg.gnn.dim_inner
            
            conv_dict = {}
            for edge_type in self.metadata[1]:
                conv_dict[edge_type] = self._get_gnn_layer()

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            if self.edge_updates:
                layer_emlps = nn.ModuleDict()
                for edge_type in self.metadata[1]:
                    edge_type = "__".join(edge_type)
                    layer_emlps[edge_type] = nn.Sequential(
                            nn.Linear(3 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                            nn.ReLU(),
                            nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                        )
                self.emlps.append(layer_emlps)
            
            if self.multi_edge_agg:
                self.edge_aggrs.append(
                    HeteroMultiEdgeAggregator(
                        n_hidden=cfg.gnn.dim_inner,
                        agg_type=self.multi_edge_agg_type,
                        data=self.data
                    )
                )

            if self.layer_norm:
                self.norms.append(nn.LayerNorm(norm_dim))
            elif self.batch_norm:
                self.norms.append(nn.BatchNorm1d(norm_dim))

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
                    edge_dim=cfg.gnn.dim_inner, towers=cfg.gnn.towers, pre_layers=1, post_layers=1,
                    divide_input=False)
        
        elif cfg.gnn.layer_type == 'RGCN':
            conv = FastRGCNConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, num_relations=15,
                            use_edge_attr=False)
        
        elif cfg.gnn.layer_type == 'RGCNE':
            conv = FastRGCNConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, num_relations=15,
                            use_edge_attr=True)
                
        else:
            raise NotImplementedError(f"{cfg.gnn.layer_type} is not implemented!")
        
        return conv

    def forward(self, batch):

        x_dict = batch.collect('x')

        for i in range(cfg.gnn.layers_mp):

            if self.multi_edge_agg:
                prev_edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
                prev_edge_attr_dict = {edge_type: batch[edge_type].edge_attr for edge_type in batch.edge_types}

                if self.layer_type in ['RGCN' ,'RGCNE']:
                    prev_currency_type_dict = {edge_type: batch[edge_type].currency_type for edge_type in batch.edge_types}

                batch = self.edge_aggrs[i](batch) 

            if self.layer_type == 'RGCN' or self.layer_type == 'RGCNE': 
                x_dict_new = self.convs[i](x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.currency_type_dict)
            else:
                x_dict_new = self.convs[i](x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # Apply normalization if enabled
            if self.layer_norm or self.batch_norm:
                x_dict_new = {
                    node_type: self.norms[i](x) for node_type, x in x_dict_new.items()
                    }

            # Apply activation and dropout
            x_dict_new = {
                node_type: self.drop(self.activation(x)) for node_type, x in x_dict_new.items()
                }

            # Apply residual connection if enabled
            if cfg.gnn.residual:
                for node_type in x_dict_new.keys():
                    x_dict[node_type] = (x_dict[node_type] + x_dict_new[node_type]) / 2
            else:
                x_dict = x_dict_new
            
            # Update edge features if enabled
            if self.edge_updates:
                if self.multi_edge_agg:
                    for edge_type in batch.edge_types:
                        prev_edge_index = prev_edge_index_dict[edge_type]
                        prev_edge_attr = prev_edge_attr_dict[edge_type]
                        src, dst = prev_edge_index

                        remapped_edge_attr = torch.index_select(batch[edge_type].edge_attr, 0, batch[edge_type].inverse_indices)
                        batch[edge_type].edge_attr = (
                                            prev_edge_attr +
                                            self.emlps[i]["__".join(edge_type)](
                                                torch.cat(
                                                    [x_dict[edge_type[0]][src], remapped_edge_attr, prev_edge_attr],
                                                    dim=-1
                                                ))
                                            ) / 2
                        # Update edge index so that in the next layer, the edge index is the same as the previous layer
                        batch[edge_type].edge_index = prev_edge_index
                        if self.layer_type == 'RGCN' or self.layer_type == 'RGCNE': 
                            # Update currency type so that in the next layer, the currency type is the same as the previous layer
                            batch[edge_type].currency_type = prev_currency_type_dict[edge_type]
                else:
                    for edge_type in batch.edge_types:
                        src, dst = batch[edge_type].edge_index
                        batch[edge_type].edge_attr = (
                                            batch[edge_type].edge_attr +
                                            self.emlps[i]["__".join(edge_type)](
                                                torch.cat(
                                                    [x_dict[edge_type[0]][src], x_dict[edge_type[0]][dst], batch[edge_type].edge_attr],
                                                    dim=-1
                                                ))
                                            ) / 2

        for node_type, x in x_dict.items():
            batch[node_type].x = x
        

        return batch
