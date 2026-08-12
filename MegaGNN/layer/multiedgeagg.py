import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import logging
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData

from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder


class BaseEdgeAggregator(nn.Module):
    """Base class for all edge aggregators"""
    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        raise NotImplementedError

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class IdentityAggregator(BaseEdgeAggregator):
    """Identity aggregator that returns input without modification"""
    def forward(self, x, index):
        return x


class SumAggregator(BaseEdgeAggregator):
    """Simple sum aggregation"""
    def forward(self, x, index):
        return scatter(x, index, dim=0, reduce='sum')


class GinAggregator(BaseEdgeAggregator):
    """GIN-style aggregation with MLP"""
    def __init__(self, n_hidden):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )

    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.nn(out)


class PnaAggregator(BaseEdgeAggregator):
    """Principal Neighbor Aggregation"""
    def __init__(self, n_hidden, deg):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.num_aggregators = len(aggregators)

        self.agg = DegreeScalerAggregation(aggregators, scalers, deg)
        self.lin = nn.Linear(len(scalers) * len(aggregators) * n_hidden, n_hidden)

    def forward(self, x, index):
        out = self.agg(x, index)
        return self.lin(out)


class AdammAggregator(BaseEdgeAggregator):
    """Adamm-style aggregation with edge transformation"""
    def __init__(self, n_hidden):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )

    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.edge_transform(out)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based aggregation"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerAggregator(BaseEdgeAggregator):
    """Transformer-based aggregation with positional encoding"""
    def __init__(self, d_model=66):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=0.05, max_len=128)
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_model*4,
            dropout=0.0,
            batch_first=True,
            norm_first=False
        )

    def forward(self, x, index, timestamps_):
        # Add timestamps for sorting
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        # Sort and prepare batch
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self._sort_wrt_time(dense_edge_feats, mask)
        
        # Apply transformer
        sorted_dense_edge_feats = self.pos_enc(sorted_dense_edge_feats.permute(1,0,2)).permute(1,0,2)
        sorted_dense_edge_feats = self.trans_enc(sorted_dense_edge_feats, src_key_padding_mask=~sorted_mask)
        sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def _sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0]
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask


class GruAggregator(BaseEdgeAggregator):
    """GRU-based aggregation"""
    def __init__(self, d_model=66):
        super().__init__()
        self.gru = nn.GRU(
            d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x, index, timestamps_):
        # Add timestamps for sorting
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        # Sort and prepare batch
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self._sort_wrt_time(dense_edge_feats, mask)

        # Apply GRU
        sorted_dense_edge_feats = self.gru(sorted_dense_edge_feats)[0]
        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def _sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0]
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask


class MultiEdgeAggregation(nn.Module):
    """
    Multi-edge aggregation module that supports various aggregation types.
    
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
            - 'genagg': Generalized aggregation
        index (torch.Tensor, optional): Node indices for PNA
        times (torch.Tensor, optional): Timestamps for temporal aggregators
    """
    def __init__(self, n_hidden=None, agg_type=None, data=None):
        super().__init__()
        self.agg_type = agg_type

        # Initialize appropriate aggregator
        if agg_type == 'identity':
            self.agg = IdentityAggregator()
        elif agg_type == 'sum':
            self.agg = SumAggregator()
        elif agg_type == 'gin':
            self.agg = GinAggregator(n_hidden=n_hidden)
        elif agg_type == 'pna':
            if not isinstance(data, HeteroData):
                uniq_index, inverse_indices = torch.unique(data.simp_edge_batch, return_inverse=True)
                d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
            else:
                uniq_index, inverse_indices = torch.unique(data.to_homogeneous().simp_edge_batch, return_inverse=True)
                d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
            deg = torch.bincount(d, minlength=1)
            self.agg = PnaAggregator(n_hidden=n_hidden, deg=deg)
        elif agg_type == 'adamm':
            self.agg = AdammAggregator(n_hidden=n_hidden)
        elif agg_type == 'transformer':
            self.agg = TransformerAggregator(d_model=n_hidden)
        elif agg_type == 'gru':
            self.agg = GruAggregator(d_model=n_hidden)
        elif agg_type == 'genagg':
            self.agg = GenAgg(f=MLPAutoencoder, jit=False)
        else:
            self.agg = IdentityAggregator()

    def forward(self, edge_index, edge_attr, simp_edge_batch, currency_type=None, timestamps_=None):
        """
        Forward pass of the multi-edge aggregation module.
        
        Args:
            edge_index (torch.Tensor): Edge indices
            edge_attr (torch.Tensor): Edge attributes
            simp_edge_batch (torch.Tensor): Batch indices for edge simplification
            currency_type (torch.Tensor, optional): Edge types based on received currency 
        Returns:
            tuple: (new_edge_index, new_edge_attr, inverse_indices)
        """
        # Get unique indices and inverse mapping
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        # simp_edge_batch = torch.index_select(_, 0, inverse_indices) : this is the way of reconstructing the simp_edge_batch
        # Compute new edge indices
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index

        # Compute new edge types
        new_currency_type = scatter(currency_type, inverse_indices, dim=0, reduce='mean') \
            if currency_type is not None and self.agg_type is not None else currency_type

        # Apply appropriate aggregation
        if self.agg_type in ['transformer', 'gru']:
            if self.times is None:
                raise ValueError("Times tensor required for temporal aggregators")
            new_edge_attr = self.agg(x=edge_attr, index=inverse_indices, timestamps_=timestamps_)
        else:
            new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)

        return new_edge_index, new_edge_attr, inverse_indices, new_currency_type

    def reset_parameters(self):
        """Reset parameters of the aggregator"""
        self.agg.reset_parameters() 