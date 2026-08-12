from functools import partial

import torch
import torch.nn as nn

from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.register import register_act



register_act('relu', nn.ReLU(inplace=cfg.mem.inplace))
register_act('gelu', nn.GELU())
register_act('elu', nn.ELU(inplace=cfg.mem.inplace))