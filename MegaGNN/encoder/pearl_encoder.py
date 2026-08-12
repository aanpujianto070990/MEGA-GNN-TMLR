import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import HeteroData
from MegaGNN.graphgym.config import cfg
from torch_geometric.utils import to_undirected

class ElementWiseGate(nn.Module):
    """Element-wise gating mechanism for skip connections."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, residual):
        gate_values = torch.sigmoid(self.gate(x))
        return gate_values * x + (1 - gate_values) * residual


class PearlEncoder(nn.Module):
    """
    GNN model with random node feature initialization for heterogeneous graphs.
    
    This model processes heterogeneous batch graphs, extracts nodes of type 'node' and 
    edges of type ('node', 'to', 'node'), generates M independent random initializations
    for node features, processes them through GIN layers, and saves the resulting 
    embeddings back to the batch object.
    
    Args:
        node_feature_dim (int): Dimension of initial random node features (default: 1)
        hidden_dim (int): Hidden dimension for GNN layers (default: 64)
        num_layers (int): Number of GNN layers (default: 3)
        M (int): Number of independent random initializations (default: 5)
        output_dim (int): Final output dimension for node embeddings (default: 64)
    """
    
    def __init__(self):
        super().__init__()
        
        self.num_layers = cfg.pearl.num_gnn_layers
        self.M = cfg.pearl.num_samples
        self.dim_pe = cfg.pearl.dim_pe
        self.undirected = cfg.pearl.use_undirected
        self.use_gating = cfg.pearl.use_gating
        
        # Input projection layer
        self.input_proj = nn.Linear(1, self.dim_pe)

        self.convs = nn.ModuleList()
        self.gates = nn.ModuleList() if self.use_gating else None
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.dim_pe, self.dim_pe*2),
                nn.BatchNorm1d(self.dim_pe*2),
                nn.ReLU(),
                nn.Linear(self.dim_pe*2, self.dim_pe)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

            if self.use_gating:
                self.gates.append(ElementWiseGate(self.dim_pe))


        
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers):
            self.bns.append(nn.BatchNorm1d(self.dim_pe))
        
        # Output projection
        self.output_proj = nn.Sequential(
                nn.Linear(self.dim_pe, 4*self.dim_pe),
                nn.BatchNorm1d(4*self.dim_pe),
                nn.ReLU(),
                nn.Linear(4*self.dim_pe, self.dim_pe)
            )
            
    
    def forward(self, batch):
        """
        Forward pass of the GNN model for heterogeneous batch graphs.
        
        Args:
            batch (HeteroData): Heterogeneous batch graph data
            
        Returns:
            HeteroData: The input batch with node embeddings added to 'node' type
        """
        # Extract node type 'node' information
        node_type = 'node'
        edge_type = ('node', 'to', 'node')
        
        device = batch[node_type].x.device if hasattr(batch[node_type], 'x') else 'cpu'
        num_nodes = batch[node_type].x.size(0)
        edge_index = batch[edge_type].edge_index

        if self.undirected:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # Initialize M random node features
        random_features = self._initialize_random_features(num_nodes, device)  # (M, num_nodes, 1)
        
        all_node_embeddings = []
        
        # Process each of the M random initializations
        for m in range(self.M):
            x = random_features[m]  # (num_nodes, 1)
            
            # Input projection
            x = self.input_proj(x)  # (num_nodes, dim_pe)

            # Pass through GNN layers with intermediate activation and normalization
            for i, conv in enumerate(self.convs):
                x_res = x
                x_new = conv(x, edge_index)
                x_new = self.bns[i](x_new)
                x_new = F.relu(x_new)
                if self.use_gating:
                    x = self.gates[i](x_new, x_res)
                else:
                    x = x_new + x_res
            
            all_node_embeddings.append(x)

        # Stack all M node embeddings and take mean across initializations
        stacked_embeddings = torch.stack(all_node_embeddings, dim=0)  # (M, num_nodes, dim_pe)
        pooled_embeddings = torch.mean(stacked_embeddings, dim=0)  # (num_nodes, dim_pe)

        # Output projection to get final node embeddings
        out = self.output_proj(pooled_embeddings)  # (num_nodes, dim_pe)
            
        batch[node_type].pearl_pe = out # (num_nodes, dim_pe)
        return batch
    
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
        random_features = torch.randn(self.M, num_nodes, 1, device=device)
        
        return random_features
    