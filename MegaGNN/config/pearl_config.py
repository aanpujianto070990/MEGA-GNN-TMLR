from yacs.config import CfgNode as CN
from MegaGNN.graphgym.register import register_config

@register_config('cfg_pearl')
def set_cfg_pearl(cfg):
    r"""This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    """
    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #
    # example argument group
    cfg.pearl = CN()
    
    cfg.pearl.enable = False
    cfg.pearl.type = 'add' # [add, concat]
    cfg.pearl.use_undirected = False
    cfg.pearl.reverse_mp = False # It must be used when use_undirected is False
    cfg.pearl.use_gating = False
    cfg.pearl.use_edge_attr = False
    cfg.pearl.hidden_dim = 64
    cfg.pearl.num_gnn_layers = 3
    cfg.pearl.num_samples = 5 # Number of independent random initializations
    cfg.pearl.dim_pe = 64 # Final output dimension for positional encodings
    cfg.pearl.which_implementation = 'original' # [original, ours]
    cfg.pearl.gnn_layer_type = "GIN" # [GIN, PNA]
    cfg.pearl.multi_edge_agg = False # [True, False]
    cfg.pearl.use_in_mpnn = False # [True, False]

    return cfg
