import json
import logging
import os
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



def train_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer
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

    time_start = time.time()
    for it,batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.device))

        pred, true = model(batch) # pred is logits 
        time_used = time.time() - time_start

        loss, pred_score = compute_loss(pred, true) # pred_score is probability, after sigmoid
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        cfg.params = params_count(model)
        logger.update_stats(true=_true,
                            pred=_pred, 
                            loss=loss.detach().cpu().item(),
                            lr=0,
                            time_used=time_used,
                            params=cfg.params, 
                            index = None
                            )
        time_start = time.time()



def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _collect_batch_graph_stats(batch) -> Dict[str, float]:
    """Extract graph-size statistics from a single PyG batch."""
    stats: Dict[str, float] = {}

    # Number of nodes
    stats['num_nodes'] = float(batch.num_nodes)

    # Number of edges (|E|)
    num_edges = batch['node', 'to', 'node'].edge_index.size(1) if batch['node', 'to', 'node'].edge_index is not None else 0
    stats['num_edges'] = float(num_edges)

    # Number of unique (i,j) pairs (|E^supp|) — the key quantity for MEGA
    if batch['node', 'to', 'node'].edge_index is not None and num_edges > 0:
        ei = batch['node', 'to', 'node'].edge_index  # [2, num_edges]
        # Encode each (src, dst) pair as a single int for fast unique count
        unique_pairs = torch.unique(ei[0] * batch.num_nodes + ei[1]).numel()
        stats['num_edge_supp'] = float(unique_pairs)
        stats['edge_multiplicity'] = stats['num_edges'] / max(stats['num_edge_supp'], 1.0)
    else:
        stats['num_edge_supp'] = 0.0
        stats['edge_multiplicity'] = 0.0

    # Number of seed/labeled edges (= transactions to classify) if present
    # In many PyG pipelines for edge classification, this is stored in edge_label_index.
    if hasattr(batch['node', 'to', 'node'], 'edge_label_index'):
        num_tx = batch['node', 'to', 'node'].edge_label_index.size(1)
        stats['num_transactions'] = float(num_tx)
    else:
        num_tx = batch['node'].batch_size
        stats['num_transactions'] = float(num_tx)

    return stats


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
    Forward-pass benchmark that also measures data loading time
    (neighbor sampling + CPU-to-GPU transfer).
    Uses CUDA synchronize for accurate GPU timings.
    Also records peak GPU memory and per-batch graph statistics.
    """
    model.eval()
    epoch_times: List[float] = []
    all_data_times: List[float] = []       # sampling + .to(device) per batch
    all_forward_times: List[float] = []    # model(batch) per batch
    all_batch_times: List[float] = []      # data + forward per batch
    all_batch_stats: List[Dict[str, float]] = []

    # Reset peak memory counter so we measure only this benchmark
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup (not timed, but warms up DataLoader workers and GPU)
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

    # Reset again after warmup so peak reflects only the timed runs
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    total_edges_timed = 0
    total_nodes_timed = 0
    total_transactions_timed = 0
    for _ in range(num_epochs):
        epoch_batch_data_times: List[float] = []
        epoch_batch_fwd_times: List[float] = []
        batches_done = 0

        # Start the data timer *before* the for-loop so the first
        # iteration's sampling time is captured.
        _synchronize_if_cuda(device)
        data_start = time.perf_counter()

        for batch in loader:
            # --- data loading time: sampling (inside DataLoader) + GPU transfer ---
            batch.split = split
            batch.to(device)
            _synchronize_if_cuda(device)
            data_end = time.perf_counter()

            # Collect graph-size stats (cheap, already on device)
            stats = _collect_batch_graph_stats(batch)
            all_batch_stats.append(stats)
            total_edges_timed += int(stats.get('num_edges', 0.0))
            total_nodes_timed += int(stats.get('num_nodes', 0.0))
            total_transactions_timed += int(stats.get('num_transactions', 0.0))

            # --- forward time ---
            fwd_start = time.perf_counter()
            _ = model(batch)
            _synchronize_if_cuda(device)
            fwd_end = time.perf_counter()

            data_t = data_end - data_start
            fwd_t = fwd_end - fwd_start

            epoch_batch_data_times.append(data_t)
            epoch_batch_fwd_times.append(fwd_t)
            all_data_times.append(data_t)
            all_forward_times.append(fwd_t)
            all_batch_times.append(data_t + fwd_t)
            batches_done += 1

            # Reset data_start for the *next* iteration's sampling measurement
            _synchronize_if_cuda(device)
            data_start = time.perf_counter()

        epoch_times.append(
            float(np.sum(epoch_batch_data_times) + np.sum(epoch_batch_fwd_times))
        )

    # ---- Aggregate timing stats ----
    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    std_epoch_s = float(np.std(epoch_times)) if epoch_times else 0.0

    avg_data_time_s = float(np.mean(all_data_times)) if all_data_times else 0.0
    avg_forward_time_s = float(np.mean(all_forward_times)) if all_forward_times else 0.0
    avg_batch_time_s = float(np.mean(all_batch_times)) if all_batch_times else 0.0
    std_batch_time_s = float(np.std(all_batch_times)) if all_batch_times else 0.0

    total_data_time_s = float(np.sum(all_data_times)) if all_data_times else 0.0
    total_forward_time_s = float(np.sum(all_forward_times)) if all_forward_times else 0.0
    total_e2e_time_s = float(np.sum(all_batch_times)) if all_batch_times else 0.0

    data_time_fraction = (total_data_time_s / total_e2e_time_s) if total_e2e_time_s > 0 else 0.0

    # ---- Peak GPU memory ----
    if device.type == 'cuda':
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    else:
        peak_mem_bytes = 0
        peak_mem_gb = 0.0

    # ---- Aggregate graph-size stats across batches ----
    def _mean_stat(key: str) -> float:
        vals = [s[key] for s in all_batch_stats if key in s]
        return float(np.mean(vals)) if vals else 0.0

    graph_stats = {
        'avg_num_nodes': _mean_stat('num_nodes'),
        'avg_num_edges': _mean_stat('num_edges'),
        'avg_num_edge_supp': _mean_stat('num_edge_supp'),
        'avg_edge_multiplicity': _mean_stat('edge_multiplicity'),
        'avg_num_transactions': _mean_stat('num_transactions'),
    }

    # Forward-only throughput (model compute only)
    fwd_edges_per_s = (float(total_edges_timed) / total_forward_time_s) if total_forward_time_s > 0 else 0.0
    fwd_transactions_per_s = (float(total_transactions_timed) / total_forward_time_s) if total_forward_time_s > 0 else 0.0

    # End-to-end throughput (data loading + forward)
    e2e_edges_per_s = (float(total_edges_timed) / total_e2e_time_s) if total_e2e_time_s > 0 else 0.0
    e2e_transactions_per_s = (float(total_transactions_timed) / total_e2e_time_s) if total_e2e_time_s > 0 else 0.0

    return {
        # ------------------------------------------------------------------
        # Headline metrics (use these in tables/plots for inference scaling)
        # ------------------------------------------------------------------
        # End-to-end transactions/sec: includes data loading (sampling + transfer).
        'throughput_transactions_per_s_e2e': e2e_transactions_per_s,
        # Forward-only transactions/sec: model compute only (no data loading).
        'throughput_transactions_per_s': fwd_transactions_per_s,
        # Peak GPU memory allocated during timed forward passes (GB).
        'peak_gpu_mem_gb': peak_mem_gb,

        # ------------------------------------------------------------------
        # Batch graph statistics (drivers of runtime; useful to explain scaling)
        # ------------------------------------------------------------------
        'avg_num_transactions': graph_stats['avg_num_transactions'],
        'avg_num_nodes': graph_stats['avg_num_nodes'],
        'avg_num_edges': graph_stats['avg_num_edges'],               # |E|
        'avg_num_edge_supp': graph_stats['avg_num_edge_supp'],       # |E^supp|
        'avg_edge_multiplicity': graph_stats['avg_edge_multiplicity'],  # mu = |E|/|E^supp|

        # ------------------------------------------------------------------
        # Per-batch timing breakdown
        # ------------------------------------------------------------------
        'avg_data_time_s': avg_data_time_s,        # Mean data loading time per batch
        'avg_forward_time_s': avg_forward_time_s,  # Mean forward-only time per batch
        'avg_batch_time_s': avg_batch_time_s,      # Mean end-to-end time per batch
        'std_batch_time_s': std_batch_time_s,      # Std of end-to-end time per batch
        'data_time_fraction': data_time_fraction,  # Fraction of total time in data loading

        # ------------------------------------------------------------------
        # Aggregate timing details
        # ------------------------------------------------------------------
        'total_data_time_s': total_data_time_s,    # Total data loading time
        'total_forward_time_s': total_forward_time_s,  # Total forward time
        'total_e2e_time_s': total_e2e_time_s,      # Total end-to-end time
        'avg_epoch_s': avg_epoch_s,                # Avg epoch time (data + forward)
        'std_epoch_s': std_epoch_s,                # Std across num_epochs repeats

        # ------------------------------------------------------------------
        # Metadata (run configuration; useful for reproducibility)
        # ------------------------------------------------------------------
        'split': split,
        'num_epochs': num_epochs,
        'num_batches_per_epoch': len(loader) if hasattr(loader, '__len__') else None,
        'total_batches': len(all_batch_times),

        # Diagnostic throughput over all sampled subgraph edges
        'throughput_edges_per_s': fwd_edges_per_s,
        'throughput_edges_per_s_e2e': e2e_edges_per_s,
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
        'Throughput (e2e): %.1f transactions/sec; %.1f edges/sec (total_e2e_time=%.3fs)',
        res['throughput_transactions_per_s_e2e'],
        res['throughput_edges_per_s_e2e'],
        res['total_e2e_time_s'],
    )
    logging.info(
        'Throughput (forward-only): %.1f transactions/sec; %.1f edges/sec (total_fwd_time=%.3fs)',
        res['throughput_transactions_per_s'],
        res['throughput_edges_per_s'],
        res['total_forward_time_s'],
    )
    logging.info(
        'Peak GPU memory: %.3f GB',
        res['peak_gpu_mem_gb'],
    )
    logging.info(
        'Per-batch breakdown: data=%.4f ms, forward=%.4f ms, total=%.4f ms '
        '(data fraction=%.1f%%)',
        res['avg_data_time_s'] * 1000.0,
        res['avg_forward_time_s'] * 1000.0,
        res['avg_batch_time_s'] * 1000.0,
        res['data_time_fraction'] * 100.0,
    )
    logging.info(
        'Avg batch graph stats: nodes=%.1f, edges(|E|)=%.1f, '
        'unique pairs(|E^supp|)=%.1f, edge multiplicity(mu)=%.2f',
        res['avg_num_nodes'],
        res['avg_num_edges'],
        res['avg_num_edge_supp'],
        res['avg_edge_multiplicity'],
    )
    logging.info(
        'Avg seed edges (transactions) per batch: %.1f',
        res['avg_num_transactions'],
    )
    logging.info(
        'Benchmark (%s): avg epoch %.4f +/- %.4f s; '
        'epochs=%d; batches/epoch=%s; total_batches=%d',
        res['split'],
        res['avg_epoch_s'],
        res['std_epoch_s'],
        res['num_epochs'],
        str(res['num_batches_per_epoch']),
        res['total_batches'],
    )

    # Persist results as JSON so external scripts can collect them
    json_path = os.path.join(cfg.run_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)
    logging.info('Benchmark results saved to %s', json_path)

@register_train('new_benckmark')
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
        num_epochs=3,
        warmup_batches=10,
    )
    logging.info('Task done, results saved in %s', cfg.run_dir)
