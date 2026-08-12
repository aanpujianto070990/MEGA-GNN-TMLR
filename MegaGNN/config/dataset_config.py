from MegaGNN.graphgym.register import register_config
from typing import Union

@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The entity to perform the task in an heterogeneous graph dataset
    cfg.dataset.task_entity = None

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    cfg.dataset.reverse_mp = False
    cfg.dataset.add_ports = False
    cfg.dataset.fraudgt_ports = False
    cfg.dataset.permute_ports = False
    cfg.dataset.port_emb_dim = 8
    cfg.dataset.full_one_edge_feats = False
    cfg.dataset.random_edge_feats = False
    cfg.dataset.new_random_each_batch = False

    cfg.dataset.rand_split = False

    cfg.dataset.synthetic_label_family = "A" # "A" or "B"
    cfg.dataset.collapse_rate = 0.0 # Ablation studies for ETH Kaggle dataset