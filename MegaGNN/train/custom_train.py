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
from MegaGNN.layer.temporal import temporal_state

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

        if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        
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


def _seed_edge_info(model: nn.Module, batch) -> Dict[str, torch.Tensor]:
    """Identifiers of the seed edges the head predicts for, in the head's order.

    Uses the same mask as `HeteroGNNEdgeHead._apply_index`. Seed edges that the
    neighbour sampler did not include in the batch get no prediction.
    """
    task = cfg.dataset.task_entity
    store = batch[task]
    inds = getattr(model.post_mp, f'{batch.split}_inds')
    mask = torch.isin(store.e_id, inds[store.input_id])
    info = {'e_id': store.e_id[mask], 'timestamp': store.timestamps[mask]}
    if 'edge_id' in store:
        info['edge_id'] = store.edge_id[mask]
    return info


@torch.no_grad()
def eval_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module,
    split: str,
    collect: bool = False
    ) -> Optional[Dict[str, np.ndarray]]:
    """Evaluate the model on a dataset.

    Args:
        logger: Logger for tracking evaluation statistics
        loader: DataLoader for evaluation data
        model: Neural network model
        collect: If True, also return per-edge identifiers, labels and scores

    Returns:
        With `collect`, a dict of arrays (e_id, timestamp, [edge_id,] y, score);
        otherwise None.
    """
    model.eval()
    time_start = time.time()
    collected = []

    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        info = _seed_edge_info(model, batch) if collect else None
        pred, true = model(batch)

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        if collect:
            info = {k: v.cpu() for k, v in info.items()}
            info['y'] = _true
            info['score'] = _pred
            collected.append(info)

        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params, index=None)
        time_start = time.time()

    if not collect:
        return None
    return {k: torch.cat([c[k] for c in collected]).numpy() for k in collected[0]}


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


@register_train('custom')
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
    if cfg.train.test_at_best:
        return train_test_at_best(loggers, loaders, model, optimizer, scheduler)

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

    # Zero-shot evaluation: Evaluate loaded checkpoint before any training
    if cfg.train.auto_resume and start_epoch > 0:
        logging.info('='*80)
        logging.info('ZERO-SHOT EVALUATION: Testing loaded checkpoint before training')
        logging.info('='*80)
        
        # Evaluate on all splits (val and test)
        for i in range(1, num_splits):
            eval_epoch(loggers[i], loaders[i], model, split=split_names[i-1])
            zero_shot_perf = loggers[i].write_epoch(start_epoch - 1)
            
            logging.info(f'Zero-shot {split_names[i-1]} performance:')
            for metric, value in zero_shot_perf.items():
                logging.info(f'  {split_names[i-1]}_{metric}: {value:.4f}')
        
        logging.info('='*80)

    # Setup wandb if enabled
    run = setup_wandb(cfg)
    if run is not None and cfg.wandb.watch_model:
        run.watch(model, log="all", log_freq=1)

    # Print model summary
    logging.info(get_model_summary(model, loaders))

    # # Training setup
    # num_splits = len(loggers)
    # split_names = ['val', 'test']
    # full_epoch_times = []
    # perf = [[] for _ in range(num_splits)]

    # Main training loop
    if cfg.train.auto_resume:
        end_epoch = start_epoch + cfg.optim.max_epoch
    else:
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
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]

                    # Log best R2 independently (argmax), even when primary
                    # model selection metric is different (e.g., MAE/F1/loss).
                    if all('r2' in vp for vp in val_perf):
                        best_r2_epoch = np.array([vp['r2'] for vp in val_perf]).argmax()
                        bstats["best/r2"] = perf[2][best_r2_epoch]['r2']

                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

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
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)

_SCORE_DTYPES = {'e_id': np.int32, 'edge_id': np.int32, 'timestamp': np.int32,
                 'y': np.int8, 'score': np.float32}


def _save_scores(scores: Dict[str, np.ndarray], split: str) -> None:
    # Compact dtypes keep a test file at a few MB (ids < 2^31, timestamps are
    # whole seconds well below 2^31, labels are 0/1).
    scores = {k: v.astype(_SCORE_DTYPES.get(k, v.dtype)) for k, v in scores.items()}
    np.savez_compressed(os.path.join(cfg.run_dir, f'scores_{split}.npz'), **scores)


def train_test_at_best(
    loggers: List[CustomLogger],
    loaders: List[DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler
) -> None:
    """Thesis protocol: model selection on validation, a single test evaluation.

    Every `train.eval_period` epochs only the validation split is evaluated.
    The checkpoint with the best validation `cfg.metric_best` is kept (ties keep
    the earlier epoch). Training stops after `cfg.optim.max_epoch` epochs, or
    earlier after `train.early_stop_patience` evaluations without improvement
    (counted only once the best validation value is > 0). The best checkpoint
    is then evaluated once on the test split.

    Besides the usual stats.json files, writes `run_summary.json` (best epoch,
    val/test metrics, timings, peak GPU memory, prediction coverage) and, with
    `train.save_scores`, per-transaction scores for val (best epoch) and test.
    """
    metric = cfg.metric_best
    patience = cfg.train.early_stop_patience
    device = torch.device(cfg.device)
    use_cuda = device.type == 'cuda'

    logging.info(get_model_summary(model, loaders))
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    best_val, best_epoch, best_val_perf, bad_evals = -np.inf, -1, None, 0
    train_times, val_times = [], []
    stopped_early = False

    for cur_epoch in range(cfg.optim.max_epoch):
        t0 = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        loggers[0].write_epoch(cur_epoch)
        train_times.append(time.perf_counter() - t0)

        if not is_eval_epoch(cur_epoch, 0):
            continue

        t0 = time.perf_counter()
        val_scores = eval_epoch(loggers[1], loaders[1], model, split='val',
                                collect=cfg.train.save_scores)
        val_perf = loggers[1].write_epoch(cur_epoch)
        val_times.append(time.perf_counter() - t0)

        if val_perf[metric] > best_val:
            best_val, best_epoch, best_val_perf, bad_evals = val_perf[metric], cur_epoch, val_perf, 0
            save_ckpt(model, optimizer, scheduler, cur_epoch)
            clean_ckpt()  # keep only the best checkpoint
            if cfg.train.save_scores:
                _save_scores(val_scores, 'val')
        elif best_val > 0:
            bad_evals += 1

        logging.info(
            f"> Epoch {cur_epoch}: train {train_times[-1]:.1f}s, val {val_times[-1]:.1f}s | "
            f"val_{metric}: {val_perf[metric]:.4f} | best: epoch {best_epoch} "
            f"(val_{metric} {best_val:.4f}) | no improvement: {bad_evals}"
            + (f"/{patience}" if patience > 0 else ""))

        if patience > 0 and bad_evals >= patience:
            stopped_early = True
            logging.info(f"Early stopping at epoch {cur_epoch}")
            break

    epochs_run = cur_epoch + 1

    # Single test evaluation with the best validation checkpoint.
    load_ckpt(model, epoch=best_epoch)
    t0 = time.perf_counter()
    test_scores = eval_epoch(loggers[2], loaders[2], model, split='test',
                             collect=cfg.train.save_scores)
    test_perf = loggers[2].write_epoch(best_epoch)
    test_time = time.perf_counter() - t0
    if cfg.train.save_scores:
        _save_scores(test_scores, 'test')

    head = model.post_mp
    run_summary = {
        'seed': cfg.seed,
        'metric_best': metric,
        'max_epoch': cfg.optim.max_epoch,
        'early_stop_patience': patience,
        'epochs_run': epochs_run,
        'stopped_early': stopped_early,
        'best_epoch': best_epoch,
        'val': best_val_perf,
        'test': test_perf,
        'params': cfg.params,
        'train_epoch_s': train_times,
        'val_eval_s': val_times,
        'test_eval_s': test_time,
        'train_iters_per_epoch': len(loaders[0]),
        'peak_gpu_mem_allocated_mb': (torch.cuda.max_memory_allocated(device) / 2**20) if use_cuda else None,
        'peak_gpu_mem_reserved_mb': (torch.cuda.max_memory_reserved(device) / 2**20) if use_cuda else None,
        'n_seed_edges': {s: int(getattr(head, f'{s}_inds').numel()) for s in ['train', 'val', 'test']},
        # Test seed edges that got a prediction (the sampler may leave a few out).
        'n_test_pred_edges': int(len(test_scores['score'])) if test_scores is not None else None,
        'batch_accumulation': cfg.optim.batch_accumulation,
        # Temporal module settings and learned parameters of the best checkpoint.
        'temporal': {'encoding': cfg.temporal.encoding, 'decay': cfg.temporal.decay,
                     'proj_init': cfg.temporal.proj_init, 'horizon_days': cfg.temporal.horizon_days,
                     **temporal_state(model)},
    }
    with open(os.path.join(cfg.run_dir, 'run_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)

    logging.info(
        f"Best epoch {best_epoch} of {epochs_run} run"
        + (" (early stopped)" if stopped_early else "")
        + f" | val_{metric}: {best_val:.4f} | test_{metric}: {test_perf[metric]:.4f}"
        + f" | avg train epoch {np.mean(train_times):.1f}s | test eval {test_time:.1f}s")

    for logger in loggers:
        logger.close()
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
    if cfg.gnn.head == 'hetero_edge_missing_rev':
        batch = add_missing_rev_edges(batch, loaders[0].loader.data)
    
    batch.to(torch.device(cfg.device))
    return summary(model, batch)
    # return None