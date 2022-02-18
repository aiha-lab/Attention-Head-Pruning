from typing import Dict, Any
import math
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


@register_scheduler("HappyInvSqrtLR")
class HappyInvSqrtLR(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_iterations: int = 0,
                 init_keep_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyInvSqrtLR, self).__init__(optimizer, warmup_iterations, init_keep_iterations, min_lr, mode,
                                             verbose=verbose)

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if initial_lr <= self.min_lr:
            return initial_lr

        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        elif self.num_iterations < self.warmup_iterations + self.init_keep_iterations:
            lr = initial_lr
        else:
            # following FairSeq implementation of InverseSquareRootSchedule.
            curr_iterations = self.num_iterations - self.init_keep_iterations
            scale = float(math.sqrt(self.warmup_iterations + 1)) / float(math.sqrt((curr_iterations + 1)))
            lr = self.min_lr + (initial_lr - self.min_lr) * min(scale, 1.0)
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            warmup_iterations=config.get("warmup_iterations", 0),
            init_keep_iterations=config.get("init_keep_iterations", 0),
            min_lr=config.get("min_lr", 1e-8),
            mode=config.get("mode", "min")
        )


HappyNoamLR = HappyInvSqrtLR
