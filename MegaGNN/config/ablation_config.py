from yacs.config import CfgNode as CN

from MegaGNN.graphgym.register import register_config


@register_config('ablation_cfg')
def set_ablation_cfg(cfg):
    cfg.ablation = CN()
    # When True, randomly permutes source node IDs within each target's
    # incoming edge set per batch, destroying source-feature coherence while
    # preserving in-degree, multiset of features, and two-stage architecture.
    cfg.ablation.random_grouping = False
