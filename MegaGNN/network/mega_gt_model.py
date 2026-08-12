import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from MegaGNN.graphgym.models import head  # noqa, register module
from MegaGNN.graphgym import register as register
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.register import register_network
from MegaGNN.graphgym.models.layer import BatchNorm1dNode
from torch_geometric.utils import (to_undirected, to_dense_batch)

from MegaGNN.layer.mega_gt_layer import MEGAGTLayer
from MegaGNN.layer.hetero_multiedgeagg import HeteroMultiEdgeAggregator
from MegaGNN.network.utils import GTPreNN
from torch_geometric.nn import (Sequential, Linear, HeteroConv, GraphConv, SAGEConv, HGTConv, GATConv)

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
            self.node_encoder = NodeEncoder(cfg.gt.dim_hidden, dataset)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.gt.dim_hidden)
            # Update dim_in to reflect the new dimension of the node features
            if self.is_hetero:
                self.dim_in = {node_type: cfg.gt.dim_hidden for node_type in dim_in}
            else:
                self.dim_in = cfg.gt.dim_hidden
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gt.dim_hidden)
            else:
                cfg.gnn.dim_edge = cfg.gt.dim_hidden
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


@register_network('MEGAGTModel')
class MEGAGTModel(torch.nn.Module):
    """
    Mega Graph Transformer Model for heterogeneous graph data.

    Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        dataset: The dataset containing the graph data.
    """
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.metadata = dataset[0].metadata() if self.is_hetero else [("node_type",), (("node_type", "edge_type", "node_type"), )]
        self.dim_h = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = register.act_dict[cfg.gt.act]
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm    = cfg.gt.l2_norm
        self.multi_edge_agg = cfg.gnn.multi_edge_agg
        self.multi_edge_agg_type = cfg.gnn.multi_edge_agg_type
        GNNHead         = register.head_dict[cfg.gt.head]

        # Initialize feature encoder
        self.encoder = FeatureEncoder(dim_in, dataset)
        self.dim_in = self.encoder.dim_in
        
        # Initialize graph transformer layers
        self.convs = nn.ModuleList()
        self.edge_aggrs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        for i in range(cfg.gt.layers):
            conv = MEGAGTLayer(self.dim_h, self.dim_h, self.dim_h, self.metadata, cfg.gt.attn_heads, 
                               layer_norm=self.layer_norm,
                               batch_norm=self.batch_norm,
                               return_attention=False)
            self.convs.append(conv)

            if self.multi_edge_agg:
                self.edge_aggrs.append(
                    HeteroMultiEdgeAggregator(
                        n_hidden=self.dim_h,
                        agg_type=self.multi_edge_agg_type,
                        data=dataset[0]
                    )
                )
                layer_emlps = nn.ModuleDict()
                for edge_type in self.metadata[1]:
                    edge_type = "__".join(edge_type)
                    layer_emlps[edge_type] = nn.Linear(3 * self.dim_h, self.dim_h)
                self.emlps.append(layer_emlps)
            

        # Initialize post-transformer head
        self.post_gt = GNNHead(self.dim_h, dim_out, dataset)


    def forward(self, batch):
        """
        Forward pass through the Mega Graph Transformer model.

        Args:
            batch: The input batch of graph data.

        Returns:
            The output of the model after processing the input batch.
        """
        # Encode features
        batch = self.encoder(batch)
        
        # Collect node and edge indices
        if isinstance(batch, HeteroData):
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
        else:
            h_dict = {self.metadata[0][0]: batch.x}
            edge_index_dict = {self.metadata[1][0]: batch.edge_index}

        # Apply input dropout
        h_dict = {node_type: self.input_drop(h_dict[node_type]) for node_type in h_dict}

        # Update batch with dropped-out features
        if isinstance(batch, HeteroData):
            for node_type in batch.node_types:
                batch[node_type].x = h_dict[node_type]
        else:
            batch.x = h_dict[self.metadata[0][0]]

        # Pass through each graph transformer layer
        for i in range(cfg.gt.layers):
            if self.multi_edge_agg:
                prev_edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
                prev_edge_attr_dict = {edge_type: batch[edge_type].edge_attr for edge_type in batch.edge_types}
                batch = self.edge_aggrs[i](batch)
            batch = self.convs[i](batch)
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
                                                [batch[edge_type[0]].x[src], remapped_edge_attr, prev_edge_attr],
                                                dim=-1
                                            ))
                                        ) / 2
                    batch[edge_type].edge_index = prev_edge_index
                    
        # Apply post-transformer head
        return self.post_gt(batch)
