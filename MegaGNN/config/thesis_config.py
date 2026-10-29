from yacs.config import CfgNode as CN

from MegaGNN.graphgym.register import register_config


@register_config('cfg_thesis')
def set_cfg_thesis(cfg):
    r"""Options for the thesis training protocol (see configs/thesis/).
    The defaults keep upstream behaviour.
    """
    # If True, evaluate only the validation split during training, keep the
    # checkpoint with the best `cfg.metric_best` on validation, and evaluate
    # the test split once with that checkpoint after training ends.
    cfg.train.test_at_best = False

    # Stop after this many evaluations without improvement of the validation
    # `cfg.metric_best` (0 disables). Only used with `train.test_at_best`.
    # Counting starts once the best validation value is > 0, so a model that
    # has not produced any positive prediction yet is never stopped.
    cfg.train.early_stop_patience = 0

    # Save per-transaction scores (scores_val.npz at the best epoch and
    # scores_test.npz) in the run directory. Only used with `train.test_at_best`.
    cfg.train.save_scores = False

    # Temporal module (MegaGNN/layer/temporal.py). Input: the edge attribute
    # `delta_t` = seconds since the sender's previous incoming transaction
    # (-1 if none), used in days. The defaults switch the module off.
    cfg.temporal = CN()
    # Encoding of delta_t added to the edge features: 'none', 'time2vec'
    # (learned frequencies and phases) or 'sinusoid' (same form and initial
    # values, frozen).
    cfg.temporal.encoding = 'none'
    # Number of periodic components k (the encoding has k + 1 dimensions).
    cfg.temporal.dim = 8
    # Initialisation of the projection of the encoding onto the edge features:
    # 'zero' (the model starts identical to the backbone) or 'default'
    # (PyTorch's Linear init; used by the first four runs, see thesis/done_runs.txt).
    cfg.temporal.proj_init = 'zero'
    # Horizon H of delta_t, in days (0 = none): delta_t_H = min(delta_t, H), and
    # senders without incoming money in the last H days (or ever) get H and a
    # recent flag of 0. The thesis configs use 2 days (decision 2026-10-29);
    # the runs before that had no horizon.
    cfg.temporal.horizon_days = 0.0
    # Shortest / longest initial period of the periodic components, in days.
    cfg.temporal.min_period_days = 1.0 / 24
    cfg.temporal.max_period_days = 16.0
    # Temporal decay: multiply each transaction's message by exp(-lambda * delta_t)
    # before the first sum aggregation (no normalisation), lambda >= 0 learned.
    cfg.temporal.decay = False
    # Initial lambda, per day.
    cfg.temporal.decay_init = 0.1
