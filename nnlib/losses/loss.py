from typing import Dict, Any
import torch
import nnlib.nn as nn


class BaseLoss(nn.BaseModule):

    def __init__(self, *, reduction: str = "mean"):
        super(BaseLoss, self).__init__()
        reduction = reduction.lower()
        if reduction not in ("none", "mean", "sum", None):
            raise ValueError(f"[ERROR:LOSS] Invalid loss reduction {reduction}.")
        if reduction is None:
            reduction = "none"
        self.reduction = reduction

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        raise NotImplementedError
