import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import MegaGNN.graphgym.register as register
from MegaGNN.graphgym.config import cfg


def compute_loss(pred: torch.Tensor, true: torch.Tensor, epoch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss and prediction score.

    Args:
        pred: Unnormalized prediction tensor
        true: Ground truth tensor
        epoch: Current epoch number (optional)

    Returns:
        Tuple containing:
        - Loss tensor
        - Normalized prediction score tensor
    """
    # Squeeze extra dimensions if present
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try custom loss functions first
    for func in register.loss_dict.values():
        result = func(pred, true, epoch)
        if result is not None:
            return result

    # Convert true to float for loss computation
    true = true.float()

    # Handle different loss functions
    if cfg.model.loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            # Multiclass classification
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        else:
            # Binary or multilabel classification
            return nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)(pred, true), torch.sigmoid(pred)
    
    elif cfg.model.loss_fun == 'mse':
        # Regression
        return nn.MSELoss(reduction=cfg.model.size_average)(pred, true), pred
    
    else:
        raise ValueError(f'Loss function {cfg.model.loss_fun} not supported')
