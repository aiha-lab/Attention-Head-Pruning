from typing import Dict, Any
import math
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


@register_scheduler("HappyExponentialLR")
class HappyExponentialLR(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 decay_factor: float = 0.999,
                 warmup_iterations: int = 0,
                 init_keep_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyExponentialLR, self).__init__(optimizer, warmup_iterations, init_keep_iterations, min_lr, mode,
                                                 verbose=verbose)
        if decay_factor >= 1.0:
            raise ValueError(f"[ERROR:SCHED] ExponentialLR decay_factor {decay_factor} >= 1.0.")
        self.decay_factor = decay_factor

    def state_dict(self) -> dict:
        d = super(HappyExponentialLR, self).state_dict()
        d["decay_factor"] = self.decay_factor
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        super(HappyExponentialLR, self).load_state_dict(state_dict)
        self.decay_factor = state_dict.get("decay_factor", 0.999)

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if initial_lr <= self.min_lr:
            return initial_lr

        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        elif self.num_iterations < self.warmup_iterations + self.init_keep_iterations:
            lr = initial_lr
        else:
            curr_iterations = self.num_iterations - self.warmup_iterations - self.init_keep_iterations
            lr = self.min_lr + (initial_lr - self.min_lr) * math.pow(self.decay_factor, curr_iterations)
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            decay_factor=config.get("decay_factor", 0.999),
            warmup_iterations=config.get("warmup_iterations", 0),
            init_keep_iterations=config.get("init_keep_iterations", 0),
            min_lr=config.get("min_lr", 1e-8),
            mode=config.get("mode", "min")
        )
