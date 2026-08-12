import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.loss import compute_loss
from MegaGNN.graphgym.register import register_train
from MegaGNN.graphgym.utils.comp_budget import params_count

from MegaGNN.logger import CustomLogger


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> int:
    """One training epoch; mirrors custom_train.train_epoch without logging (one step per batch)."""
    model.train()
    optimizer.zero_grad()
    batches = 0
    for batch in loader:
        batch.split = 'train'
        batch.to(device)

        pred, true = model(batch)
        loss, _pred_score = compute_loss(pred, true)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batches += 1

    _synchronize_if_cuda(device)
    return batches


def train_benchmark(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    num_epochs: int = 5,
    warmup_batches: int = 10,
    split: str = 'train',
) -> Dict[str, Any]:
    """
    Wall-clock training benchmark: forward + backward + optimizer (and scheduler step).
    One optimizer step per batch; optional grad clipping from cfg.
    """
    # Warmup (untimed)
    model.train()
    optimizer.zero_grad()
    warmup_done = 0
    for batch in loader:
        if warmup_done >= warmup_batches:
            break
        batch.split = 'train'
        batch.to(device)
        pred, true = model(batch)
        loss, _ = compute_loss(pred, true)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        cfg.params = params_count(model)
        warmup_done += 1
    _synchronize_if_cuda(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times: List[float] = []
    total_batches = 0

    for _ in range(num_epochs):
        _synchronize_if_cuda(device)
        t0 = time.perf_counter()
        n = _train_one_epoch(model, loader, optimizer, device)
        t1 = time.perf_counter()
        epoch_times.append(float(t1 - t0))
        total_batches += n

    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    std_epoch_s = float(np.std(epoch_times)) if epoch_times else 0.0

    if device.type == 'cuda':
        peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    else:
        peak_mem_bytes = 0
        peak_mem_gb = 0.0

    num_batches_per_epoch = len(loader) if hasattr(loader, '__len__') else None

    return {
        'avg_epoch_s': avg_epoch_s,
        'std_epoch_s': std_epoch_s,
        'peak_gpu_mem_gb': peak_mem_gb,
        'peak_mem_bytes': peak_mem_bytes,
        'num_epochs': num_epochs,
        'total_batches': total_batches,
        'num_batches_per_epoch': num_batches_per_epoch,
        'split': split,
    }


def run_train_benchmark(
    model: nn.Module,
    loaders: List[DataLoader],
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    num_epochs: int = 5,
    warmup_batches: int = 10,
) -> None:
    """Use train loader (index 0); log and save JSON under cfg.run_dir."""
    if not loaders:
        raise IndexError('No loaders provided')
    device = torch.device(cfg.device)
    res = train_benchmark(
        model=model,
        loader=loaders[0],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        warmup_batches=warmup_batches,
        split='train',
    )

    logging.info(
        'Train benchmark: avg_epoch_s=%.4f +/- %.4f s; peak_gpu_mem_gb=%.4f',
        res['avg_epoch_s'],
        res['std_epoch_s'],
        res['peak_gpu_mem_gb'],
    )
    logging.info(
        'Train benchmark: epochs=%d; batches/epoch=%s; total_batches=%d',
        res['num_epochs'],
        str(res['num_batches_per_epoch']),
        res['total_batches'],
    )

    json_path = os.path.join(cfg.run_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)
    logging.info('Train benchmark results saved to %s', json_path)


@register_train('train_benchmark')
def custom_train(
    loggers: List[CustomLogger],
    loaders: List[DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
) -> None:
    logging.info('Running train_benchmark (avg_epoch_s, peak_gpu_mem_gb)')
    run_train_benchmark(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=1,
        warmup_batches=10,
    )
    logging.info('Task done, results saved in %s', cfg.run_dir)
