from yacs.config import CfgNode as CN
from MegaGNN.graphgym.register import register_config

@register_config('cfg_mega')
def set_cfg_mega(cfg):
    r"""This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    """
    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.mega = CN()

    
