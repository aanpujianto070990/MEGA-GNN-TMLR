import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from MegaGNN.graphgym import register as register
from MegaGNN.graphgym.config import cfg



class SparseAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hidden = cfg.gt.dim_hidden
        self.num_heads = cfg.gt.attn_heads
        self.head_dim = cfg.gt.dim_hidden // cfg.gt.attn_heads
        # Define the matrices associated with the Sparse Attention mechanism (as defined in the SGFormer model)
        # Define the matrices associated with the Sparse Attention mechanism
        self.Wk = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)
        self.Wq = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)
        self.Wv = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)


    def forward(self, query_input, source_input):

        edge_bucket_ids = self._compute_sparse_buckets(source_input)

        sorted_ids = torch.argsort(edge_bucket_ids)
        ln_source_input_sorted = source_input[sorted_ids]
        ln_query_input_sorted = query_input[sorted_ids]
        # clusters sorted in ascending order
        edge_bucket_ids_sorted = edge_bucket_ids[sorted_ids]

        # Project
        qs = self.Wq(ln_query_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
        ks = self.Wk(ln_source_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
        vs = self.Wv(ln_source_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]

        # Group by cluster
        unique, counts = edge_bucket_ids_sorted.unique_consecutive(return_counts=True)
        # Find the ending index of each cluster by cumulatively summing the counts of each cluster
        cluster_ptr = counts.cumsum(0)
        # Find the starting index of each cluster
        cluster_start = torch.cat([torch.tensor([0], device=source_input.device), cluster_ptr[:-1]])

        output = torch.zeros_like(qs)
        for i in range(len(unique)):
            s, e = cluster_start[i].item(), cluster_ptr[i].item()
            q = qs[s:e].transpose(0, 1)  # [H, N, d_h]
            k = ks[s:e].transpose(0, 1)
            v = vs[s:e].transpose(0, 1)

            # Intra-cluster scaled-dot product attention
            attn_out = F.scaled_dot_product_attention(q, k, v)  # [H, N, d_h]
            output[s:e] = attn_out.transpose(0, 1) # [N, H, d_h]

        # Merge heads and unsort
        output = output.reshape(output.size(0), -1) # [N, H*d_h=d]
        
        final_output = torch.zeros_like(output, device=query_input.device)
        final_output[sorted_ids] = output

        return final_output
    
    def _compute_sparse_buckets(self, attr):
        if cfg.gt.sparse_clustering == "lsh":
            # According to the Reformerpaper
            num_bits = 16
            edge_attr_attn = torch.nn.functional.normalize(attr, p=2, dim=-1)
            d = edge_attr_attn.size(-1)
            R = torch.randn(d, num_bits, device=cfg.device)
            hash_bits = (edge_attr_attn @ R) > 0  # [M, num_bits] boolean tensor
            hash_bits = hash_bits.int() # convert to int tensor
            powers = 2 ** torch.arange(num_bits, device=cfg.device)
            bucket_ids = (hash_bits * powers).sum(dim=-1)
            
            num_buckets = max(1, math.ceil(edge_attr_attn.size(0)/cfg.gt.cluster_size))
            bucket_ids = bucket_ids % num_buckets
            return bucket_ids  # [M]
        else:
            num_edges = attr.shape[0]
            num_buckets = max(1, math.ceil(num_edges / cfg.gt.cluster_size))
            return torch.randint(low=0, high=num_buckets, size=(num_edges,), device=cfg.device)

class LinearAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hidden = cfg.gt.dim_hidden
        self.num_heads = cfg.gt.attn_heads
        self.head_dim = cfg.gt.dim_hidden // cfg.gt.attn_heads
        # Define the matrices associated with the Linear Attention mechanism (as defined in the SGFormer model)
        self.Wk = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)
        self.Wq = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)
        self.Wv = nn.Linear(self.n_hidden, self.head_dim * self.num_heads, device=cfg.device)

    def forward(self, query_input, source_input):

        # Peform the forward pass considering the Linear Attention Mechanism (as defined in the SGFormer model)
        # Construct the Q and K matrices
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
        vs = self.Wv(source_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
        
        # Normalize the Q and K matrices
        qs = qs / torch.norm(qs, p=2) # [N, H, d_h]
        ks = ks / torch.norm(ks, p=2) # [N, H, d_h]
        N = qs.shape[0]

        # Compute the all pair attentive representations
        # Numerator
        kvs = torch.einsum("nhd,nhk->hdk", ks, vs) # [H, d_h, d_h]
        attention_num = torch.einsum("nhd,hdd->nhd", qs, kvs) # [N, H, d_h]
        attention_num += vs * N # [N, H, d]

        # Denominator
        all_ones = torch.ones([ks.shape[0]], device=ks.device)
        ks_sum = torch.einsum("nhd,n->hd", ks, all_ones) # [H, d_h]
        attention_normalizer = torch.einsum("nhd,hd->nh", qs, ks_sum) # [N, H]

        # Attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)) # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer, device=attention_normalizer.device) * N
        attn_output = attention_num / attention_normalizer # [N, H, d_h]
    
        final_output = attn_output.reshape(attn_output.size(0), -1) # [N, H*d_h]

        return final_output
        

class GlobalAttentionLayer(torch.nn.Module):
    """
    Attention-based encoder which computes contextualized representations of the edge in the sampled sub-graph.
    """
    def __init__(self):
        super().__init__()
        self.n_hidden = cfg.gt.dim_hidden
        self.num_heads = cfg.gt.attn_heads
        self.head_dim = cfg.gt.dim_hidden // cfg.gt.attn_heads
        self.dropout = cfg.gt.attn_dropout
        self.attention_mechanism = cfg.gt.attn_mechanism

        if self.attention_mechanism == "linear":
            self.self_attn = LinearAttentionLayer()

        elif self.attention_mechanism == "full":
            self.self_attn = torch.nn.MultiheadAttention(self.n_hidden, self.num_heads, 
                                                         dropout=self.dropout, 
                                                         batch_first=True
                                                        )
            # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_hidden, nhead=self.num_heads,    dim_feedforward=4*self.n_hidden, dropout=self.dropout, activation=F.gelu, batch_first=True, norm_first=True, device=cfg.device)

        else:
            self.self_attn = SparseAttentionLayer()
    
    def forward(self, query_input, source_input, edge_bucket_ids=None):
        if self.attention_mechanism in ["linear", "full", "sparse"]:
            out = self.self_attn(query_input, source_input)
            return out
        else:
            raise ValueError("Invalid attention mechanism specified. Possible alternatives: linear, full or sparse.")

            
    



