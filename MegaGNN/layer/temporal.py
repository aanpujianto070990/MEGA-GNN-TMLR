"""Temporal module for the thesis (plug-in for Multi-GIN+EU and MEGA-GIN).

Input is the edge attribute `delta_t`: seconds since the sender of the
transaction last received money (strictly earlier), -1 if it never did
(see `causal_delta_t` in MegaGNN/datasets/aml_dataset.py). It is used in days.

Two independent components, switched on through `cfg.temporal`:
- an encoding of delta_t (Time2Vec, or the same form with frozen parameters)
  added to the encoded edge features;
- a decay weight w = exp(-lambda * delta_t) that multiplies each
  transaction's message before the first sum aggregation.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GINEConv

SECONDS_PER_DAY = 86400.0


def delta_t_days(delta_t: Tensor) -> Tuple[Tensor, Tensor]:
    """(delta_t in days with missing history set to 0, mask of transactions with history)."""
    has_prev = delta_t >= 0
    return delta_t.clamp(min=0) / SECONDS_PER_DAY, has_prev


class TimeEncoder(nn.Module):
    r"""Time2Vec (Kazemi et al., 2019) of a scalar time t:
    [w_0 t + p_0, sin(w_1 t + p_1), ..., sin(w_k t + p_k)].

    The periodic frequencies start at periods spaced log-uniformly between
    `min_period` and `max_period` (days), phases at 0, and w_0 = 1. With
    `learnable=False` the same values are kept fixed (the fixed sinusoidal
    baseline), so the two variants differ only in whether the frequencies and
    phases are learned.
    """

    def __init__(self, k: int, learnable: bool = True,
                 min_period: float = 1.0 / 24, max_period: float = 16.0):
        super().__init__()
        periods = torch.logspace(math.log10(max_period), math.log10(min_period), k)
        w = torch.cat([torch.ones(1), 2 * math.pi / periods])
        p = torch.zeros(k + 1)
        if learnable:
            self.w = nn.Parameter(w)
            self.p = nn.Parameter(p)
        else:
            self.register_buffer('w', w)
            self.register_buffer('p', p)
        self.out_dim = k + 1

    def forward(self, t: Tensor) -> Tensor:
        z = t.unsqueeze(-1) * self.w + self.p
        return torch.cat([z[:, :1], torch.sin(z[:, 1:])], dim=-1)


class TemporalEdgeEncoding(nn.Module):
    """Adds Proj([TimeEncoder(delta_t), has_prev]) to the encoded edge features.

    Adding a projection after the linear edge encoder is the same as
    concatenating the time features to the raw edge features before it.
    One projection per edge type (forward and reverse edges), like the edge
    encoder itself.

    With `zero_init` the projections start at zero, so the model starts out
    identical to the backbone and learns to use the time features gradually.
    Without it, the unbounded linear Time2Vec term (delta_t up to ~18 days)
    perturbed the normalised edge features from the first step, and Multi-GIN
    needed many more epochs before it started to learn.
    """

    def __init__(self, edge_types, dim_out: int, k: int, learnable: bool,
                 min_period: float, max_period: float, zero_init: bool = True):
        super().__init__()
        self.time_enc = TimeEncoder(k, learnable, min_period, max_period)
        self.proj = nn.ModuleDict({
            '__'.join(et): nn.Linear(self.time_enc.out_dim + 1, dim_out) for et in edge_types
        })
        if zero_init:
            for lin in self.proj.values():
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, batch):
        for et in batch.edge_types:
            t, has_prev = delta_t_days(batch[et].delta_t)
            feat = torch.cat([self.time_enc(t), has_prev.float().unsqueeze(-1)], dim=-1)
            batch[et].edge_attr = batch[et].edge_attr + self.proj['__'.join(et)](feat)
        return batch


class TemporalDecay(nn.Module):
    """w = exp(-lambda * delta_t[days]) per transaction, lambda = exp(log_lambda) >= 0.

    Transactions whose sender has no earlier incoming transaction get w = 1.
    At lambda = 0 every weight is 1 and the backbone is recovered.
    """

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))

    @property
    def lam(self) -> Tensor:
        return self.log_lambda.exp()

    def forward(self, delta_t: Tensor) -> Tensor:
        t, has_prev = delta_t_days(delta_t)
        return torch.where(has_prev, torch.exp(-self.lam * t), torch.ones_like(t))


class DecayGINEConv(GINEConv):
    """GINEConv whose messages ReLU(x_j + e_ji) are multiplied by an optional
    per-edge weight before the sum aggregation.

    Reuses GINEConv's parameters (nn, eps, edge projection) but computes the
    messages and the sum directly instead of through `propagate()`: recent PyG
    versions generate `propagate()` from GINEConv's own message signature, and
    a subclass inherits that version, which rejects the extra `edge_weight`.
    With `edge_weight=None` the result equals GINEConv's.
    """

    def forward(self, x, edge_index, edge_attr=None, edge_weight: Optional[Tensor] = None, size=None):
        x_src, x_dst = (x, x) if isinstance(x, Tensor) else x
        src, dst = edge_index
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        msg = (x_src[src] + edge_attr).relu()
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        n_dst = x_dst.size(0) if x_dst is not None else (size[1] if size is not None else x_src.size(0))
        out = msg.new_zeros(n_dst, msg.size(-1)).index_add_(0, dst, msg)
        if x_dst is not None:
            out = out + (1 + self.eps) * x_dst
        return self.nn(out)


def temporal_state(model: nn.Module) -> Dict:
    """Learned temporal parameters, for run_summary.json."""
    state = {}
    gnn = getattr(model, 'gnn', None)
    if getattr(gnn, 'decay', None) is not None:
        state['lambda_per_day'] = float(gnn.decay.lam)
    enc = getattr(model, 'temporal_enc', None)
    if enc is not None:
        w = enc.time_enc.w.detach().cpu()
        state['time2vec_linear_w'] = float(w[0])
        state['time2vec_periods_days'] = [float(2 * math.pi / abs(v)) if v != 0 else None for v in w[1:]]
    return state
