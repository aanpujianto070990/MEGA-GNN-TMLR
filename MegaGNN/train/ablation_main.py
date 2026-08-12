"""
Ablation train loop: identical to custom_train but applies
apply_random_group_control(batch) before every model forward call.

Register train mode: 'ablation_main'
Enabled via YAML:  train.mode: ablation_main
                   ablation.random_grouping: True
"""
import logging
import time
from typing import List, Optional, Any

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
from MegaGNN.train.custom_train import setup_wandb
from MegaGNN.train.ablation_utils import apply_random_group_control
from MegaGNN.train.utils import add_missing_rev_edges


def _train_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    batch_accumulation: int,
) -> None:
    model.train()
    optimizer.zero_grad()

    time_start = time.time()
    for it, batch in enumerate(loader):
        batch.split = 'train'
        batch = apply_random_group_control(batch)
        batch.to(torch.device(cfg.device))

        pred, true = model(batch)
        time_used = time.time() - time_start

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)

        loss.backward()

        if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value
                )
            optimizer.step()
            optimizer.zero_grad()

        cfg.params = params_count(model)
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time_used,
            params=cfg.params,
            index=None,
        )
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def _eval_epoch(
    logger: CustomLogger,
    loader: DataLoader,
    model: nn.Module,
    split: str,
) -> None:
    model.eval()
    time_start = time.time()

    for batch in loader:
        batch.split = split
        batch = apply_random_group_control(batch)
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)

        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            index=None,
        )
        time_start = time.time()


def _get_model_summary(model, loaders):
    batch = next(iter(loaders[0]))
    batch.split = 'train'
    if cfg.gnn.head == 'hetero_edge_missing_rev':
        batch = add_missing_rev_edges(batch, loaders[0].loader.data)
    batch.to(torch.device(cfg.device))
    return summary(model, batch)


@register_train('ablation_main')
def ablation_main(
    loggers: List[CustomLogger],
    loaders: List[DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
) -> None:
    """Training pipeline for the random-grouping ablation experiment.

    Identical to custom_train except apply_random_group_control() is called on
    every batch before the model forward pass (for both train and eval splits).
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, epoch=cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    run = setup_wandb(cfg)
    if run is not None and cfg.wandb.watch_model:
        run.watch(model, log='all', log_freq=1)

    logging.info(_get_model_summary(model, loaders))

    if cfg.train.auto_resume:
        end_epoch = start_epoch + cfg.optim.max_epoch
    else:
        end_epoch = cfg.optim.max_epoch

    for cur_epoch in range(start_epoch, end_epoch):
        start_time = time.perf_counter()

        _train_epoch(
            loggers[0], loaders[0], model, optimizer, scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                _eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        full_epoch_times.append(time.perf_counter() - start_time)

        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ''
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(
                    np.array([vp[m] for vp in val_perf]), cfg.metric_agg
                )()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {'best/epoch': best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f'best/{s}_loss'] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f'best/{s}_{m}'] = perf[i][best_epoch][m]
                            run.summary[f'best_{s}_perf'] = perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f'best/{s}_{x}'] = perf[i][best_epoch][x]

                    if all('r2' in vp for vp in val_perf):
                        best_r2_epoch = np.array(
                            [vp['r2'] for vp in val_perf]
                        ).argmax()
                        bstats['best/r2'] = perf[2][best_r2_epoch]['r2']

                    run.log(bstats, step=cur_epoch)
                    run.summary['full_epoch_time_avg'] = np.mean(full_epoch_times)
                    run.summary['full_epoch_time_sum'] = np.sum(full_epoch_times)

            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()

            logging.info(
                f'> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s '
                f'(avg {np.mean(full_epoch_times):.1f}s) | '
                f'Best so far: epoch {best_epoch}\t'
                f'train_loss: {perf[0][best_epoch]["loss"]:.4f} {best_train}\t'
                f'val_loss: {perf[1][best_epoch]["loss"]:.4f} {best_val}\t'
                f'test_loss: {perf[2][best_epoch]["loss"]:.4f} {best_test}'
            )

    logging.info(f'Avg time per epoch: {np.mean(full_epoch_times):.2f}s')
    logging.info(f'Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h')

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)
