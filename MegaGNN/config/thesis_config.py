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
