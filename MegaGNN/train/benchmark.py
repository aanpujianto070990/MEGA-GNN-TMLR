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
    
    for batch in loader:
        batch.split = split

        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

@torch.no_grad()
def forward_benchmark(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    warmup_batches: int = 10,
    split: str = 'val'
) -> Dict[str, Any]:
    """
    Minimal forward-pass benchmark (excludes data transfer).
    Uses CUDA synchronize for accurate GPU timings.
    """
    model.eval()
    epoch_times: List[float] = []
    all_batch_times: List[float] = []

    # Warmup (not timed)
    warmup_done = 0
    for batch in loader:
        if warmup_done >= warmup_batches:
            break
        batch.split = split
        batch.to(device)
        _synchronize_if_cuda(device)
        _ = model(batch)
        _synchronize_if_cuda(device)
        warmup_done += 1

    # Timed runs
    for _ in range(num_epochs):
        epoch_batch_times: List[float] = []
        for batch in loader:
            batch.split = split
            batch.to(device)

            _synchronize_if_cuda(device)
            start_t = time.perf_counter()
            _ = model(batch)
            _synchronize_if_cuda(device)
            end_t = time.perf_counter()

            t = end_t - start_t
            epoch_batch_times.append(t)
            all_batch_times.append(t)

        epoch_times.append(float(np.sum(epoch_batch_times)))

    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    avg_batch_s = float(np.mean(all_batch_times)) if all_batch_times else 0.0

    return {
        'avg_epoch_s': avg_epoch_s,
        'avg_batch_s': avg_batch_s,
        'num_epochs': num_epochs,
        'num_batches_per_epoch': len(loader) if hasattr(loader, '__len__') else None,
        'total_batches': len(all_batch_times),
        'split': split,
    }

def run_forward_benchmark(
    model: nn.Module,
    loaders: List[DataLoader],
    split: str = 'val',
    num_epochs: int = 5,
    warmup_batches: int = 10,
) -> None:
    """
    Pick loader by split and log simple stats.
    Assumes loaders: train=0, val=1, test=2 (if present).
    """
    split_to_idx: Dict[str, int] = {'train': 0, 'val': 1, 'test': 2}
    if split not in split_to_idx:
        raise ValueError(f'Unsupported split: {split}')
    idx = split_to_idx[split]
    if idx >= len(loaders):
        raise IndexError(f'Loader for split {split} not provided')

    device = torch.device(cfg.device)
    res = forward_benchmark(
        model=model,
        loader=loaders[idx],
        device=device,
        num_epochs=num_epochs,
        warmup_batches=warmup_batches,
        split=split,
    )

    logging.info(
        'Forward benchmark (%s): avg epoch %.4fs; avg batch %.4fms; epochs=%d; batches/epoch=%s; total_batches=%d',
        res['split'],
        res['avg_epoch_s'],
        res['avg_batch_s'] * 1000.0,
        res['num_epochs'],
        str(res['num_batches_per_epoch']),
        res['total_batches'],
    )

@register_train('benckmark')
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
    if cfg.train.auto_resume:
        # We do finetuning here so optimizer and scheduler are not needed to be loaded.
        start_epoch = load_ckpt(model, epoch=cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)


    # Training setup
    num_splits = len(loggers)
    split_names = [ 'val','test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    # Optional minimal forward-pass benchmark
    run_forward_benchmark(
        model=model,
        loaders=loaders,
        split='test',
        num_epochs=10,
        warmup_batches=10,
    )
    logging.info('Task done, results saved in %s', cfg.run_dir)
