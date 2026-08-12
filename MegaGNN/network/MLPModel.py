import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder

from torch_geometric.data import HeteroData
from torch_geometric.nn.models import MLP
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

    
@register_network('MLPModel')
class MLPModel(torch.nn.Module):
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

        # Create 4-layer MLP using torch.nn.Linear
        self.mlp_layer1 = nn.Linear(cfg.gnn.dim_inner, 128)
        self.mlp_layer2 = nn.Linear(128, 128)
        self.mlp_layer3 = nn.Linear(128, 64)
        self.mlp_layer4 = nn.Linear(64, cfg.gnn.dim_inner)
        
        # Dropout layers
        self.mlp_dropout = nn.Dropout(cfg.gnn.dropout)
        
        # Activation function
        self.mlp_activation = nn.ReLU()
        # Initialize post-message passing head
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h_total, dim_out, dataset)

    def forward(self, batch):
        # Encode features
        batch = self.encoder(batch)
        
        # Apply MLP to edge features only
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            # Apply input dropout to edge features
            edge_attr = self.input_drop(batch.edge_attr)
            
            # Pass edge features through 4-layer MLP
            edge_attr = self.mlp_layer1(edge_attr)
            edge_attr = self.mlp_activation(edge_attr)
            edge_attr = self.mlp_dropout(edge_attr)
            
            edge_attr = self.mlp_layer2(edge_attr)
            edge_attr = self.mlp_activation(edge_attr)
            edge_attr = self.mlp_dropout(edge_attr)
            
            edge_attr = self.mlp_layer3(edge_attr)
            edge_attr = self.mlp_activation(edge_attr)
            edge_attr = self.mlp_dropout(edge_attr)
            
            edge_attr = self.mlp_layer4(edge_attr)
            
            # Update batch with processed edge features
            batch.edge_attr = edge_attr
        elif hasattr(batch, 'edge_attr_dict'):
            # Handle heterogeneous graphs - apply MLP to each edge type
            for edge_type, edge_attr in batch.edge_attr_dict.items():
                if edge_attr is not None:
                    edge_attr = self.input_drop(edge_attr)
                    
                    # Pass edge features through 4-layer MLP
                    edge_attr = self.mlp_layer1(edge_attr)
                    edge_attr = self.mlp_activation(edge_attr)
                    edge_attr = self.mlp_dropout(edge_attr)
                    
                    edge_attr = self.mlp_layer2(edge_attr)
                    edge_attr = self.mlp_activation(edge_attr)
                    edge_attr = self.mlp_dropout(edge_attr)
                    
                    edge_attr = self.mlp_layer3(edge_attr)
                    edge_attr = self.mlp_activation(edge_attr)
                    edge_attr = self.mlp_dropout(edge_attr)
                    
                    edge_attr = self.mlp_layer4(edge_attr)
                    
                    batch.edge_attr_dict[edge_type] = edge_attr

        return self.post_mp(batch)

