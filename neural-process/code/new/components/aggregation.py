from enum import Enum

import torch


class Aggregation(Enum):
    MEAN = (lambda x, dim: torch.mean(x, dim=dim),)
    SUM = (lambda x, dim: torch.sum(x, dim=dim),)
    MAX = (lambda x, dim: torch.max(x, dim=dim),)
    MIN = (lambda x, dim: torch.min(x, dim=dim),)
