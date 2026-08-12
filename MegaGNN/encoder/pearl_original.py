'''
This code is from:
https://github.com/ehejin/Pearl-PE

This is the original implementation of R-PEARL.
'''

import torch
from torch import NoneType, nn
from typing import List, Callable
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import PNAConv
from torch_geometric.utils import get_laplacian, to_dense_adj, degree
from torch_geometric.data import Data
from MegaGNN.graphgym.config import cfg
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected


from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential
from torch.utils.data import DataLoader

from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from .pearl_encoder import ElementWiseGate


class MLP(nn.Module):
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, dropout_prob: float, use_bias=True
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, dropout_prob)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=use_bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features
    
class MLPLayer(nn.Module):
    """Simplified MLP layer with linear transformation, batch norm, activation, and dropout."""
    
    def __init__(self, in_dims: int, out_dims: int, dropout_prob: float) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        self.bn = nn.BatchNorm1d(out_dims)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear -> BatchNorm -> ReLU -> Dropout
        """
        X = self.fc(X)
        
        # Handle batch normalization for different tensor dimensions
        if X.dim() == 3:  # [batch, seq, features]
            X = X.transpose(1, 2)  # [batch, features, seq]
            X = self.bn(X)
            X = X.transpose(1, 2)  # [batch, seq, features]
        else:  # [batch, features]
            X = self.bn(X)
            
        X = self.activation(X)
        X = self.dropout(X)
        return X
    


class PNA(MessagePassing):
    '''
    - Be careful, in this PNA implementation, the tower is set to 1. If you set it other than 1, the output will be wrong.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        train_norm: bool = False,
        flow: str = "source_to_target",
        **kwargs,
    ):

        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, flow = flow, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, 2*self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        
        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = torch.cat([x, out], dim=-1)
        out = self.post_nns[0](out)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            # edge_attr = edge_attr.view(-1, 1, self.F_in)
            # edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j], dim=-1) + edge_attr
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = self.pre_nns[0](h) 
        return hs

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')

    @staticmethod
    def get_degree_histogram(loader: DataLoader) -> Tensor:
        r"""Returns the degree histogram to be used as input for the :obj:`deg`
        argument in :class:`PNAConv`.
        """
        deg_histogram = torch.zeros(1, dtype=torch.long)
        for data in loader:
            deg = degree(data.edge_index[1], num_nodes=data.num_nodes,
                         dtype=torch.long)
            deg_bincount = torch.bincount(deg, minlength=deg_histogram.numel())
            deg_histogram = deg_histogram.to(deg_bincount.device)
            if deg_bincount.numel() > deg_histogram.numel():
                deg_bincount[:deg_histogram.size(0)] += deg_histogram
                deg_histogram = deg_bincount
            else:
                assert deg_bincount.numel() == deg_histogram.numel()
                deg_histogram += deg_bincount

        return deg_histogram




class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP, flow: str = "source_to_target") -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow=flow, node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True) #torch.empty(1), requires_grad=True)
        self.mlp = mlp
        self.lin_edge = nn.Linear(cfg.pearl.hidden_dim, cfg.pearl.hidden_dim)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, laplacian=False,
                edge_attr: OptTensor = None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        if laplacian == 'L':
            edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=X.size(0))
            laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', laplacian, X)   # [N_sum, ***, D_in]

            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z)  
        elif laplacian == 'RW':
            adj = to_dense_adj(edge_index).squeeze(0)  # [N_sum, N_sum]
            deg = degree(edge_index[0], num_nodes=X.size(0), dtype=torch.float)  # [N_sum]
            deg_inv = 1.0 / deg  # Inverse of the degree
            deg_inv[deg_inv == float('inf')] = 0  # Handle division by zero for isolated nodes
            deg_inv_diag = torch.diag(deg_inv)  # [N_sum, N_sum]
            random_walk = torch.matmul(adj, deg_inv_diag)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', random_walk, X)  # [N_sum, *, D_in]
            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z)       # [N_sum, ***, D_out]

        S = self.propagate(edge_index, X=X, edge_attr=edge_attr)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor, edge_attr: OptTensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)
            X_j = X_j + edge_attr
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims



class GNN(nn.Module):
    """GNN model that perfors message passing with configurable layers and optional batch normalization."""
    
    def __init__(
        self, 
        n_layers: int, 
        in_dims: int, 
        hidden_dims: int, 
        out_dims: int, 
        create_mlp: Callable[[int, int], MLP],
        bn: bool = False, 
        laplacian=None,
        data=None
    ) -> None:
        super().__init__()
        
        self.bn = bn
        self.laplacian = laplacian
        self.data = data
        self.use_gating = cfg.pearl.use_gating
        
        if bn:
            self.batch_norms = nn.ModuleList()
        if self.use_gating:
            self.gates = nn.ModuleList()
        
        self.first_proj = nn.Linear(in_dims, hidden_dims)
        self.final_proj = nn.Linear(hidden_dims, out_dims)

        # Create intermediate layers
        self.layers = nn.ModuleList()
        if cfg.pearl.reverse_mp:
            self.rev_layers = nn.ModuleList()
            self.rev_proj = nn.ModuleList()
        if cfg.pearl.use_edge_attr:
            self.edge_updater = nn.ModuleList()
        
        if cfg.pearl.multi_edge_agg:
            self.edge_agg = nn.ModuleList()

        for i in range(n_layers):

            if cfg.pearl.gnn_layer_type == 'GIN':
                layer = GINLayer(create_mlp(hidden_dims, hidden_dims))
                if cfg.pearl.reverse_mp:
                    rev_layer = GINLayer(create_mlp(hidden_dims, hidden_dims), flow="target_to_source")
            
            else:
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                if not isinstance(self.data, HeteroData):
                    d = degree(self.data.edge_index[1], dtype=torch.long)
                else:
                    d = degree(self.data.to_homogeneous().edge_index[1], dtype=torch.long)
                deg = torch.bincount(d, minlength=1)

                layer = PNA(in_channels=hidden_dims, out_channels=hidden_dims,
                            aggregators=aggregators, 
                            scalers=scalers, 
                            deg=deg)

                if cfg.pearl.reverse_mp:
                    rev_layer = PNA(in_channels=hidden_dims, out_channels=hidden_dims,
                            aggregators=aggregators, 
                            scalers=scalers, 
                            deg=deg,
                            flow="target_to_source")

            self.layers.append(layer)
            if cfg.pearl.reverse_mp:
                self.rev_layers.append(rev_layer)
                self.rev_proj.append(nn.Linear(2*hidden_dims, hidden_dims))

            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
            
            if self.use_gating:
                self.gates.append(ElementWiseGate(hidden_dims))
            
            if cfg.pearl.use_edge_attr:
                self.edge_updater.append(nn.Linear(3*hidden_dims, hidden_dims))
            
            if cfg.pearl.multi_edge_agg:
                self.edge_agg.append(MultiEdgeAggrMinimal(n_hidden=hidden_dims))
                
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_attr: OptTensor = None, simp_edge_batch=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        X = self.first_proj(X)
        for i, layer in enumerate(self.layers):
            X0 = X

            if cfg.pearl.multi_edge_agg:
                prev_edge_index = edge_index
                prev_edge_attr = edge_attr
                edge_index, edge_attr, inverse_indices = self.edge_agg[i](edge_index, edge_attr, simp_edge_batch)

            if cfg.pearl.gnn_layer_type == 'GIN':
                X = layer(X, edge_index, laplacian=self.laplacian, edge_attr=edge_attr)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
                if cfg.pearl.reverse_mp:
                    X_rev = self.rev_layers[i](X0, edge_index, laplacian=self.laplacian, edge_attr=edge_attr)
                    X = self.rev_proj[i](torch.cat([X, X_rev], dim =-1))
            else:
                X = layer(X, edge_index)
                if cfg.pearl.reverse_mp:
                    X_rev = self.rev_layers[i](X0, edge_index)
                    X = self.rev_proj[i](torch.cat([X, X_rev], dim =-1))
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                
                if X.ndim == 3:
                    X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                else:
                    X = self.batch_norms[i](X)
            if self.use_gating:
                X = self.gates[i](X, X0)
            else:
                X = X + X0
            
            if edge_attr is not None:
                if cfg.pearl.multi_edge_agg:
                    src, dst = prev_edge_index
                    remapped_edge_attr = torch.index_select(edge_attr, 0, inverse_indices)
                    
                    edge_attr = (prev_edge_attr + self.edge_updater[i](torch.cat([X[src], remapped_edge_attr, prev_edge_attr], dim=-1))) / 2
                    edge_index = prev_edge_index
                else:
                    src, dst = edge_index
                    edge_attr = (edge_attr + self.edge_updater[i](torch.cat([X[src], X[dst], edge_attr], dim=-1))) / 2
        
        return self.final_proj(X)                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims
    
def get_lap(n, edge_index):
    L_edge_index, L_values = get_laplacian(edge_index, normalization="sym", num_nodes=n)  
    L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
    return L

'''
    Basic graph filtering used in all experiments of our paper.
'''
def filter(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK

class OriginalPearlPE(nn.Module):
    """PEARL positional encoder with graph filtering and MLP processing."""
    
    def __init__(self, k=1, mlp_nlayers=1, dataset=None) -> None:
        super().__init__()
        self.k = k
        self.num_samples = cfg.pearl.num_samples
        mlp_out = cfg.pearl.dim_pe
        mlp_hid = mlp_out
        self.num_layers = cfg.pearl.num_gnn_layers
        self.undirected = cfg.pearl.use_undirected
        
        self.mlp_nlayers = mlp_nlayers
        self.sample_aggr = GINSampleAggregator(
            n_layers=self.num_layers, in_dims=mlp_hid, hidden_dims=cfg.pearl.hidden_dim, out_dims=mlp_hid, bn=True, data=dataset[0])
        
        # Create MLP layers if needed
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert mlp_hid == mlp_out, "For single layer MLP, hidden and output dims must be equal"
                
            self.layers = nn.ModuleList([
                nn.Linear(
                    k if i == 0 else mlp_hid, 
                    mlp_hid if i < mlp_nlayers - 1 else mlp_out, 
                    bias=True
                ) for i in range(mlp_nlayers)
            ])
            
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(mlp_hid if i < mlp_nlayers - 1 else mlp_out, track_running_stats=True) 
                for i in range(mlp_nlayers)
            ])
            
            self.activation = nn.ReLU()
        else:
            self.layers = None
            self.norms = None
            self.activation = None
        
        if cfg.pearl.use_edge_attr:
            self.edge_encoder = nn.Linear(1, cfg.pearl.hidden_dim)

    def forward(self, batch: torch.Tensor, final=True) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        node_type = 'node'
        edge_type = ('node', 'to', 'node')
        
        device = batch[node_type].x.device if hasattr(batch[node_type], 'x') else 'cpu'
        num_nodes = batch[node_type].x.size(0)
        edge_index = batch[edge_type].edge_index

        if self.undirected:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        
        if cfg.pearl.multi_edge_agg:
            simp_edge_batch = batch[edge_type].simp_edge_batch
        else:
            simp_edge_batch = None

        Lap = get_lap(num_nodes, edge_index)
        W = self._initialize_random_features(num_nodes, device)  # (num_nodes, M)

        if cfg.pearl.use_edge_attr:
            num_edges = edge_index.size(1)
            We = self._initialize_random_features(num_edges, device).unsqueeze(-1)
            We = self.edge_encoder(We)
             # (num_edges, M, 1)
        else:
            We = None

        W_list = []
        for lap, w in zip(Lap.unsqueeze(0), W.unsqueeze(0)):
            output = filter(lap, w, self.k)   # output [NxMxK]
            if self.mlp_nlayers > 0:
                for layer, bn in zip(self.layers, self.norms):
                    output = output.transpose(0, 1)
                    output = layer(output)
                    output = bn(output.transpose(1,2)).transpose(1,2)
                    output = self.activation(output)
                    output = output.transpose(0, 1)
            W_list.append(output) 
        
        PE = self.sample_aggr(W_list, edge_index, final=final, edge_attr=We, simp_edge_batch=simp_edge_batch)   # [N_sum, D_pe]
        batch[node_type].pearl_pe = PE # (num_nodes, dim_pe)

        return batch


    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims

    def _initialize_random_features(self, num_nodes, device='cpu'):
        """
        Initialize M independent random node feature vectors.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            device (str): Device to create tensors on
        
        Returns:
            torch.Tensor: Shape (M, num_nodes, 1)
        """
        # Generate M independent random vectors from Gaussian distribution
        random_features = torch.randn(num_nodes, self.num_samples, device=device)
        
        return random_features


class GINSampleAggregator(nn.Module):
    """GIN-based sample aggregator for positional encoding."""
    
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, bn: bool = False, data=None) -> None:
        super().__init__()
        
        # Create a simple MLP factory function
        def create_mlp(in_dim: int, out_dim: int) -> MLP:
            return MLP(
                n_layers=3, 
                in_dims=in_dim, 
                hidden_dims=hidden_dims, 
                out_dims=out_dim, 
                dropout_prob=0.0, 
                use_bias=True
            )
        
        self.gnn = GNN(
            n_layers=n_layers, 
            in_dims=in_dims, 
            hidden_dims=hidden_dims, 
            out_dims=out_dims, 
            create_mlp=create_mlp,
            bn=bn, 
            laplacian=None,
            data = data
        )
        
        
    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, mean=False, final=True, edge_attr=None, simp_edge_batch=None) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """ 
        W = torch.cat(W_list, dim=0)  
        PE = self.gnn(W, edge_index, edge_attr=edge_attr, simp_edge_batch=simp_edge_batch)  
        PE = (PE).mean(dim=1)
        return PE              

    @property
    def out_dims(self) -> int:
        return self.gnn.out_dims


from torch_scatter import scatter

class MultiEdgeAggrMinimal(nn.Module):
    def __init__(self, n_hidden: int) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.nn = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )
    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, simp_edge_batch: torch.Tensor) -> torch.Tensor:

        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean')
        new_edge_attr = self.nn(scatter(edge_attr, inverse_indices, dim=0, reduce='sum'))

        return new_edge_index, new_edge_attr, inverse_indices
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
