from typing import Dict, Any
import math
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


@register_scheduler("HappyCosineLR")
class HappyCosineLR(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 max_iterations: int,
                 warmup_iterations: int = 0,
                 init_keep_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyCosineLR, self).__init__(optimizer, warmup_iterations, init_keep_iterations, min_lr, mode,
                                            verbose=verbose)
        if warmup_iterations + init_keep_iterations > max_iterations:
            raise ValueError(f"[ERROR:SCHED] CosineLR scheduler warmup + init_keep > max.")
        self.max_iterations = max_iterations

    def state_dict(self) -> dict:
        d = super(HappyCosineLR, self).state_dict()
        d["max_iterations"] = self.max_iterations
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        super(HappyCosineLR, self).load_state_dict(state_dict)
        self.max_iterations = state_dict.get("max_iterations")

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if initial_lr <= self.min_lr:
            return initial_lr

        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        elif self.num_iterations < self.warmup_iterations + self.init_keep_iterations:
            lr = initial_lr
        elif self.num_iterations >= self.max_iterations:
            lr = self.min_lr
        else:
            curr_iterations = self.num_iterations - self.warmup_iterations - self.init_keep_iterations
            max_iterations = self.max_iterations - self.warmup_iterations - self.init_keep_iterations

            lr = self.min_lr + 0.5 * (initial_lr - self.min_lr) * (
                    1 + math.cos(math.pi * curr_iterations / max_iterations))
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            max_iterations=config.get("max_iterations"),
            warmup_iterations=config.get("warmup_iterations", 0),
            init_keep_iterations=config.get("init_keep_iterations", 0),
            min_lr=config.get("min_lr", 1e-8),
            mode=config.get("mode", "min")
        )
