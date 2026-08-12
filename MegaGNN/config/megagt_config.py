from MegaGNN.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_megagt')
def set_cfg_gt(cfg):
    # Positional encodings argument group
    cfg.megagt = CN()
    cfg.megagt.cluster_size = 10000
    
