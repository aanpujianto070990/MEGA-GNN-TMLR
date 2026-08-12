import logging
import time
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.nn import summary

from MegaGNN.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)

from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.loss import compute_loss
from MegaGNN.graphgym.register import register_train
from MegaGNN.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from MegaGNN.graphgym.utils.comp_budget import params_count

from MegaGNN.utils import make_wandb_name, cfg_to_dict, flatten_dict
from MegaGNN.logger import CustomLogger

from .utils import add_missing_rev_edges

def train_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    batch_accumulation: int
) -> None:
    """Train the model for one epoch.
    
    Args:
        logger: Logger for tracking training statistics
        loader: DataLoader for training data
        model: Neural network model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        batch_accumulation: Number of batches to accumulate gradients before updating
    """
    model.train()
    optimizer.zero_grad()

    for it,batch in enumerate(loader):
        batch.split = 'train'
        torch.save(batch, f'/mnt/data/AML/Medium-HI/train/batch_{it}.pt')
        if it > 10:
            break


@torch.no_grad()
def eval_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module, 
    split: str
    ) -> None:
    """Evaluate the model on a dataset.
    
    Args:
        logger: Logger for tracking evaluation statistics
        loader: DataLoader for evaluation data
        model: Neural network model
    """
    model.eval()
        
    for it, batch in enumerate(loader):
        torch.save(batch, f'/mnt/data/AML/Medium-HI/{split}/batch_{it}.pt')
        if it > 10:
            break

@register_train('save')
def custom_train(
    loggers: List[CustomLogger],
    loaders: List[DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler
) -> None:
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers for different data splits (train/val/test)
        loaders: List of data loaders for different splits
        model: Neural Network model to train
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    Features:
        - Automatic checkpoint loading and saving
        - Weights & Biases integration for experiment tracking
        - Performance metrics tracking across splits
        - Configurable evaluation and checkpointing frequencies
        - Support for best model checkpointing
    """
    start_epoch = 0

    # Training setup
    num_splits = len(loggers)
    split_names = ['val', 'test']

    train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                cfg.optim.batch_accumulation)

    for i in range(1, num_splits):
        eval_epoch(loggers[i], loaders[i], model, split=split_names[i-1])
