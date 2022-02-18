from typing import Dict, Any
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


@register_scheduler("HappyConstantLR")
class HappyConstantLR(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_iterations: int = 0,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyConstantLR, self).__init__(optimizer, warmup_iterations, 0, 1e-8, mode, verbose=verbose)

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if initial_lr <= self.min_lr:
            return initial_lr

        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        else:
            lr = initial_lr
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            warmup_iterations=config.get("warmup_iterations", 0),
            mode=config.get("mode", "min")
        )
