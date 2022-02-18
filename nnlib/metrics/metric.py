from typing import Dict, Any
import torch

from nnlib.nn.modules.module import BaseModule


class BaseMetric(BaseModule):

    def __init__(self):
        super(BaseMetric, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        raise NotImplementedError
