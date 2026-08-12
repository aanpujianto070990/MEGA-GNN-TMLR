import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_scatter import scatter_max
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import degree
import math
from MegaGNN.graphgym.config import cfg
import MegaGNN.graphgym.register as register
from MegaGNN.layer.multiedgeagg import (IdentityAggregator, SumAggregator, GinAggregator, 
                                        PnaAggregator, AdammAggregator, TransformerAggregator, 
                                        GruAggregator, GenAgg)
from genagg.MLPAutoencoder import MLPAutoencoder
from MegaGNN.train.ablation_utils import group_indices_from_edge_index


class GTAggregator(nn.Module):
    def __init__(self, n_hidden=None):
        super().__init__()
  
        self.n_hidden = n_hidden
        self.num_heads = cfg.gnn.multi_edge_agg_gt_num_heads
        self.edge_gate = cfg.gnn.multi_edge_agg_gt_edge_gate

        self.activation = register.act_dict[cfg.gt.act]

        self.q_lin = nn.Linear(n_hidden, n_hidden)
        self.k_lin = nn.Linear(n_hidden, n_hidden)
        self.v_lin = nn.Linear(n_hidden, n_hidden)
        self.g_lin = nn.Linear(n_hidden, n_hidden)
        
        self.norm1_edge = nn.LayerNorm(n_hidden)
        self.norm2_ffn = nn.LayerNorm(n_hidden)

        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)
        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)

        self.ff_linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.ff_linear2 = nn.Linear(n_hidden * 2, n_hidden)

        # artificial node embedding
        self.art_node_emb = nn.Parameter(torch.empty(1, n_hidden))
        # Apply Xavier uniform initialization
        nn.init.xavier_uniform_(self.art_node_emb)

    
    def forward(self, edge_index, edge_attr, simp_edge_batch, num_nodes, unique_index_):
        num_edges = edge_index.shape[1]

        # # Create a mapping dictionary from artificial node indices to new destination IDs, as if we are creating a bipartite graph.
        # # Each parallel edge group gets mapped to a new unique node ID starting from num_nodes+1
        # # Example: if we have parallel edges pointing to nodes [12, 12, 12, 18, 18] and num_nodes = 1000, the mapping would be {12: 1001, 18: 1002}
        # artificial_node_mapping = {val.item(): num_nodes+idx for idx, val in enumerate(unique_index_)}

        # # Create new destination nodes by mapping each edge's artificial node to its new destination ID
        # # This effectively creates new unique nodes for each group of parallel edges
        # new_dest_nodes = torch.tensor([artificial_node_mapping[x.item()] for x in simp_edge_batch], device=cfg.device)

        # Much more faster version of the above two line.
        new_dest_nodes = torch.arange(num_nodes, num_nodes + unique_index_.shape[0], device=cfg.device)
        index_map = torch.zeros(unique_index_.max() + 1, dtype=torch.long, device=cfg.device)
        index_map[unique_index_] = new_dest_nodes  # Map old indices to new artificial nodes
        # Efficiently map `simp_edge_batch` using the precomputed index map
        new_dest_nodes = index_map[simp_edge_batch]
        
        # Create new edge indices by stacking the source nodes and new destination nodes
        new_edge_index = torch.stack([edge_index[0], new_dest_nodes], dim=0)

        # Pre-normalization 
        edge_attr = self.norm1_edge(edge_attr)

        # Compute query and key for each edge
        H, D = self.num_heads, self.n_hidden // self.num_heads# Apply appropriate aggregation
        L = num_nodes + unique_index_.shape[0]
        q = self.q_lin(self.art_node_emb).expand(edge_attr.shape[0], -1) # query comes from learnable artificial node embedding.
        k = self.k_lin(edge_attr)
        v = self.v_lin(edge_attr)
        q = q.view(-1, H, D)
        k = k.view(-1, H, D)
        v = v.view(-1, H, D)

        # transpose to get dimensions h * sl * d_model
        q = q.transpose(0,1)
        k = k.transpose(0,1)
        v = v.transpose(0,1)

        if self.edge_gate:
            edge_gate = self.g_lin(edge_attr)
            edge_gate = edge_gate.view(-1, H, D)
            edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

        edge_attr = edge_attr.view(-1, H, D)
        edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)
    
        src_nodes, dst_nodes = new_edge_index
        # Step 1:Compute attention scores
        edge_scores = q * k
        
        if self.edge_gate:
            v = v * F.sigmoid(edge_gate)

        edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # num_heads * num_edges
        edge_scores = torch.clamp(edge_scores, min=-5, max=5)

        expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head

        # Step 2: Calculate max for each destination node per head using scatter_max
        max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L) # This broadcasts the destination node indices across all 8 feature dimensions, making it possible to perform the scatter operation independently for each head.
        max_scores = max_scores.gather(1, expanded_dst_nodes)

        # Step 3: Exponentiate scores and sum
        exp_scores = torch.exp(edge_scores - max_scores)
        sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
        sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores) # (dim, index, src)

       # Step 4: Apply softmax
        edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
        edge_scores = edge_scores.unsqueeze(-1)
        edge_scores = self.dropout_attn(edge_scores)
        saved_scores = edge_scores

        out = torch.zeros((H, L, D), device=q.device)

        '''
        There is different ways of doing skip connection. Here we add the original edge features to the values and then aggregation is performed. 
        We can also skip connection after the aggregation. But in this case, we need to duplicate aggregated features (artificial nodes) for each multi edge
        then perform skip connection and again we need to take the average.
        without skip connection:
            out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * v)
        '''

        out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * (v+edge_attr))
        out = out.transpose(0,1).contiguous().view(-1, H * D)[num_nodes:] # All of the destionation node_ids are after number of nodes. (For artificial ndoes)

        out = out + self._ff_block(out)
        
        return out

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


class HeteroMultiEdgeAggregator(nn.Module):
    """
    Multi-edge aggregation module for heterogeneous graphs.
    
    Args:
        n_hidden (int): Hidden dimension size
        agg_type (str): Type of aggregation to use. Options:
            - 'identity': No aggregation
            - 'sum': Simple sum aggregation
            - 'gin': GIN-style aggregation
            - 'pna': Principal Neighbor Aggregation
            - 'adamm': Adamm-style aggregation
            - 'transformer': Transformer-based aggregation
            - 'gru': GRU-based aggregation
        metadata: Metadata of the dataset, containing node and edge types.
    """
    def __init__(self, n_hidden=None, agg_type='sum', data=None):
        super().__init__()
        self.agg_type = agg_type
        self.n_hidden = n_hidden
        self.metadata = data.metadata()

        # Create a ModuleList of aggregators for each edge type
        self.aggregators = nn.ModuleList()
        for edge_type in self.metadata[1]:
            if agg_type == 'identity':
                self.aggregators.append(IdentityAggregator())
            elif agg_type == 'sum':
                self.aggregators.append(SumAggregator())
            elif agg_type == 'gin':
                self.aggregators.append(GinAggregator(n_hidden=n_hidden))
            elif agg_type == 'pna':
                uniq_index, inverse_indices = torch.unique(data[edge_type].simp_edge_batch, return_inverse=True)
                d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
                deg = torch.bincount(d, minlength=1)
                self.aggregators.append(PnaAggregator(n_hidden=n_hidden, deg=deg))
            elif agg_type == 'adamm':
                self.aggregators.append(AdammAggregator(n_hidden=n_hidden))
            elif agg_type == 'transformer':
                self.aggregators.append(TransformerAggregator(d_model=n_hidden))
            elif agg_type == 'gru':
                self.aggregators.append(GruAggregator(d_model=n_hidden))
            elif agg_type == 'genagg':
                self.aggregators.append(GenAgg(f=MLPAutoencoder, jit=False))
            elif agg_type == 'gt_aggregator':
                self.aggregators.append(GTAggregator(n_hidden))
            else:
                self.aggregators.append(IdentityAggregator())

    def forward(self, data: HeteroData):
        """
        Perform multi-edge aggregation on a heterogeneous graph.
        Args:
            data (HeteroData): The input heterogeneous graph data.
        Returns:
            HeteroData: The graph data with aggregated edge attributes.
        """
        # Iterate over each edge type and corresponding aggregator
        for i, edge_type in enumerate(data.edge_types):
            edge_index = data[edge_type].edge_index
            edge_attr = data[edge_type].edge_attr
            simp_edge_batch = data[edge_type].simp_edge_batch

            # Get unique indices and inverse mapping.
            # When random_grouping is enabled, derive grouping from the shuffled
            # (src, dst) pairs in edge_index instead of the precomputed simp_edge_batch.
            if cfg.ablation.random_grouping:
                group_ids = group_indices_from_edge_index(edge_index)
                unique_index_, inverse_indices = torch.unique(group_ids, return_inverse=True)
            else:
                unique_index_, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)

            # Compute new edge indices
            new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index

            # Apply appropriate aggregation
            if self.agg_type in ['gt_aggregator']:
                new_edge_attr = self.aggregators[i](edge_index, edge_attr, simp_edge_batch, data.num_nodes, unique_index_)    
            else:
                new_edge_attr = self.aggregators[i](x=edge_attr, index=inverse_indices)

            # Update the graph with new edge attributes
            data[edge_type].edge_index = new_edge_index
            data[edge_type].edge_attr = new_edge_attr
            data[edge_type].inverse_indices = inverse_indices

            if cfg.gnn.layer_type == 'RGCN' or cfg.gnn.layer_type == 'RGCNE':
                currency_type = data[edge_type].currency_type
                new_currency_type = scatter(currency_type, inverse_indices, dim=0, reduce='mean') if self.agg_type is not None else currency_type
                data[edge_type].currency_type = new_currency_type

        return data

    def reset_parameters(self):
        """Reset parameters of all aggregators"""
        for aggregator in self.aggregators:
            aggregator.reset_parameters()