from typing import List, Dict, Any
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


@register_scheduler("HappyStepLR")
class HappyStepLR(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 steps: List[int],
                 multiply_factor: float = 0.1,
                 warmup_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyStepLR, self).__init__(optimizer, warmup_iterations, 0, min_lr, mode, verbose=verbose)
        if steps[0] == 0:
            steps = steps[1:]
        if steps != sorted(steps):
            raise ValueError(f"[ERROR:SCHED] StepLR step {steps} not sorted.")
        if warmup_iterations >= steps[0]:
            raise ValueError("[ERROR:SCHED] Warmup steps should be smaller than first step[0].")
        self.steps = steps
        self.multiply_factor = multiply_factor

    def state_dict(self) -> dict:
        d = super(HappyStepLR, self).state_dict()
        d["steps"] = self.steps
        d["multiply_factor"] = self.multiply_factor
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        super(HappyStepLR, self).load_state_dict(state_dict)
        self.steps = state_dict.get("steps")
        self.multiply_factor = state_dict.get("multiply_factor")

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        else:
            curr_iterations = self.num_iterations - self.warmup_iterations
            lr = initial_lr
            for s in self.steps:
                if curr_iterations >= s:
                    lr *= self.multiply_factor
            lr = max(lr, self.min_lr)
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            steps=config.get("steps"),
            multiply_factor=config.get("multiply_factor", 0.1),
            warmup_iterations=config.get("warmup_iterations", 0),
            min_lr=config.get("min_lr", 1e-8),
            mode=config.get("mode", "min")
        )
