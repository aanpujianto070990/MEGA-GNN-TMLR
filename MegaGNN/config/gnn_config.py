from MegaGNN.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gnn')
def set_gnn_cfg(cfg):
    """
    Set GNN specific configurations that are not in the default config
    """
    # ----------------------------------------------------------------------- #
    # GNN options
    # ----------------------------------------------------------------------- #
    cfg.gnn = CN()

    # Prediction head. Use cfg.dataset.task by default
    cfg.gnn.head = 'default'

    # Number of layers before message passing
    cfg.gnn.layers_pre_mp = 0

    # Number of layers for message passing
    cfg.gnn.layers_mp = 2

    # Number of layers after message passing
    cfg.gnn.layers_post_mp = 0

    # Hidden layer dim. Automatically set if train.auto_match = True
    cfg.gnn.dim_inner = -1  # Updated to match dim_hidden
    
    cfg.gnn.dim_hidden = 64  # Updated to match dim_hidden

    # Type of graph conv: GINE, GenAgg, PNA, generalconv, gcnconv, sageconv, gatconv, ...
    cfg.gnn.layer_type = 'GINE'

    # Stage type: 'stack', 'skipsum', 'skipconcat'
    cfg.gnn.stage_type = 'stack'

    # How many layers to skip each time
    cfg.gnn.skip_every = 1

    # Whether use batch norm
    cfg.gnn.batchnorm = False

    # Whether to use layer normalization
    cfg.gnn.layer_norm = False

    # Activation
    cfg.gnn.act = 'relu'

    # Dropout
    cfg.gnn.dropout = 0.0

    # Input dropout rate (separate from regular dropout)
    cfg.gnn.input_dropout = 0.0

    # Aggregation type: add, mean, max
    # Note: only for certain layers that explicitly set aggregation type
    # e.g., when cfg.gnn.layer_type = 'generalconv'
    cfg.gnn.agg = 'add'

    # Message passing flow: source_to_target or target_to_source
    cfg.gnn.flow = 'source_to_target'

    # Normalize adj
    cfg.gnn.normalize_adj = False

    # Message direction: single, both
    cfg.gnn.msg_direction = 'single'

    # Whether add message from node itself: none, add, cat
    cfg.gnn.self_msg = 'concat'

    # Number of attention heads
    cfg.gnn.attn_heads = 1

    # After concat attention heads, add a linear layer
    cfg.gnn.attn_final_linear = False

    # After concat attention heads, add a linear layer
    cfg.gnn.attn_final_linear_bn = False

    # Normalize after message passing
    cfg.gnn.l2norm = True

    # randomly use fewer edges for message passing
    cfg.gnn.keep_edge = 0.5

    # clear cached feature_new
    cfg.gnn.clear_feature = True
    
    # Whether to update edge features during message passing
    cfg.gnn.edge_updates = True
    cfg.gnn.use_edge_feats = True
        
    # Edge aggregation type for multi-edge graphs
    # Available options:
    # - 'identity': No aggregation
    # - 'sum': Simple sum aggregation
    # - 'gin': GIN-style aggregation with MLP
    # - 'pna': Principal Neighbor Aggregation
    # - 'adamm': Adamm-style aggregation with edge transformation
    # - 'transformer': Transformer-based aggregation with positional encoding
    # - 'gru': GRU-based aggregation
    # - 'genagg': Generalized aggregation
    cfg.gnn.multi_edge_agg = False
    cfg.gnn.multi_edge_agg_type = 'sum'
    cfg.gnn.towers = 5

    cfg.gnn.multi_edge_agg_gt_num_heads = 2
    cfg.gnn.multi_edge_agg_gt_edge_gate = False
    
    # Whether to use residual connections
    cfg.gnn.residual = True
    
    return cfg 