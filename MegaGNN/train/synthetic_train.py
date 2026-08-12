'''
This file is used to train the model on the synthetic pattern dataset.
author: Cagri Bilgi
date: 2026-02-12
'''

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

import wandb

from MegaGNN.utils import make_wandb_name, cfg_to_dict, flatten_dict
from MegaGNN.logger import CustomLogger

from .utils import add_missing_rev_edges


def add_edge_features(batch):
        
    batch['node', 'to', 'node'].edge_attr = torch.randn(batch['node', 'to', 'node'].edge_index.shape[1], cfg.share.dim_in['__'.join(('node', 'to', 'node'))]).float()
    if ('node', 'rev_to', 'node') in batch.edge_types:
        batch['node', 'rev_to', 'node'].edge_attr = torch.randn(batch['node', 'rev_to', 'node'].edge_index.shape[1], cfg.share.dim_in['__'.join(('node', 'rev_to', 'node'))]).float()
    return batch

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

    time_start = time.time()
    for it,batch in enumerate(loader):
        optimizer.zero_grad()
        batch.split = 'train'

        if cfg.dataset.new_random_each_batch:
            batch = add_edge_features(batch)

        batch.to(torch.device(cfg.device))

        pred, true = model(batch) # pred is logits 
        time_used = time.time() - time_start

        loss, pred_score = compute_loss(pred, true) # pred_score is probability, after sigmoid
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        loss.backward()
        optimizer.step()
        
        cfg.params = params_count(model)
        logger.update_stats(true=_true,
                            pred=_pred, 
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time_used,
                            params=cfg.params, 
                            index = None
                            )
        time_start = time.time()
    scheduler.step()


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
    time_start = time.time()
    
    for batch in loader:
        batch.split = split

        if cfg.dataset.new_random_each_batch:
            batch = add_edge_features(batch)
            
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params, index=None)
        time_start = time.time()


def setup_wandb(cfg) -> Optional[Any]:
    """Initialize Weights & Biases logging if enabled."""
    if not cfg.wandb.use:
        return None
    
    try:
        import wandb
    except ImportError:
        raise ImportError('WandB is not installed.')
    
    wandb_name = cfg.wandb.name if cfg.wandb.name else make_wandb_name(cfg)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=wandb_name,
        dir=cfg.wandb.save_dir
    )
    run.config.update(cfg_to_dict(cfg))
    return run


@register_train('synthetic_train')
def synthetic_train(
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

    logging.info('Start from epoch %s', start_epoch)

    # Setup wandb if enabled
    run = setup_wandb(cfg)
    if run is not None and cfg.wandb.watch_model:
        run.watch(model, log="all", log_freq=1)

    F1_COLUMNS = ["best_epoch", "deg-in", "deg-out", "fan-in", "fan-out",
                  "C2", "C3", "C4", "C5", "C6", "S-G", "B-C", "macro_f1"]
    if cfg.wandb.use:
        f1_table = wandb.Table(columns=F1_COLUMNS)

    # Training setup
    num_splits = len(loggers)
    split_names = [ 'val','test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    # Print model summary
    logging.info(get_model_summary(model, loaders))
    end_epoch = cfg.optim.max_epoch

    for cur_epoch in range(start_epoch, end_epoch):
        start_time = time.perf_counter()
        
        # Training phase
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                   cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        # Evaluation phase
        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i-1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        full_epoch_times.append(time.perf_counter() - start_time)

        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        # Log to wandb if enabled
        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]}"
                else:
                    # Note: For some datasets it is too expensive to compute the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

                    if 'f1' in perf[2][best_epoch]:
                        f1_row = [best_epoch] + perf[2][best_epoch]['f1'] + [perf[2][best_epoch].get(m, 0)]
                        f1_table.add_data(*f1_row)

            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
                
            # Log epoch stats
            logging.info(
                    f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                    f"(avg {np.mean(full_epoch_times):.1f}s) | "
                    f"Best so far: epoch {best_epoch}\t"
                    f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                    f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                    f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
                )

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    
    # Cleanup
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.log({"test_f1_table": f1_table})
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)

def get_model_summary(model, loaders):
    """Generate a summary of the model architecture and parameters.
    
    Args:
        model: The neural network model to summarize
        loaders: List of data loaders, where the first loader is used for getting a sample batch
        
    Returns:
        str: A string containing the model summary
    """
    batch = next(iter(loaders[0]))
    batch.split = 'train'
    
    batch.to(torch.device(cfg.device))
    return summary(model, batch)
    # return None