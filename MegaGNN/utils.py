import os, logging
from typing import List

import torch
from torch import Tensor
from torch_geometric.utils import degree
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import scatter
from yacs.config import CfgNode

from MegaGNN.graphgym.config import (cfg, makedirs_rm_exist)
from MegaGNN.graphgym.optimizer import OptimizerConfig
from MegaGNN.optimizer.extra_optimizers import ExtendedSchedulerConfig

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag, gpu_index=-1):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    if gpu_index != -1:
        run_name += f'-gpu{gpu_index}'
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format + "-" + cfg.dataset.name

    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ['gnn', 'MegaGNNModel']:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == 'GTModel':
        model_name = f"GT.{cfg.gt.layer_type}"
    
    if cfg.dataset.reverse_mp:
        model_name += "+RMP"

    if cfg.dataset.add_ports:
        model_name += "+Ports"

    if cfg.train.add_ego_id:
        model_name += "+EGO"
    
    if cfg.gnn.multi_edge_agg:
        model_name += f"+Multi-Edge({cfg.gnn.multi_edge_agg_type})"
    
    if cfg.gnn.head == 'hetero_edge_missing_rev':
        model_name += "+eq7&11"

    # Compose wandb run name.
    name = f"{dataset_name} | {model_name}"

    return name


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict
    

def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result