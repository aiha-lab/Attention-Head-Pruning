from typing import Dict, Any
import math
import torch

from nnlib.optim.lr_scheduler import register_scheduler
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler
from nnlib.utils.print_utils import print_log


@register_scheduler("HappyReduceLROnPlateau")
class HappyReduceLROnPlateau(BaseScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 multiply_factor: float = 0.1,
                 patience: int = 1,
                 warmup_iterations: int = 0,
                 init_keep_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:
        super(HappyReduceLROnPlateau, self).__init__(optimizer, warmup_iterations, init_keep_iterations, min_lr, mode,
                                                     verbose=verbose)
        self.multiply_factor = multiply_factor
        self.patience = max(patience, 1)
        self._drop_count = 0

    def state_dict(self) -> dict:
        d = super(HappyReduceLROnPlateau, self).state_dict()
        d["multiply_factor"] = self.multiply_factor
        d["patience"] = self.patience
        d["drop_count"] = self._drop_count
        d["patience_count"] = self._patience_count
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        super(HappyReduceLROnPlateau, self).load_state_dict(state_dict)
        self.multiply_factor = state_dict.get("multiply_factor", 0.1)
        self.patience = state_dict.get("patience", 1)
        self._drop_count = state_dict.get("drop_count", 0)
        self._patience_count = state_dict.get("patience_count", 0)

    def update_best(self, criterion_value) -> bool:
        """Update best validation criterion and return if the value is updated."""
        if self.best is None:
            self.best = criterion_value
            self._drop_count = 0
            self._patience_count = 0
            s = f"...... best set, {self.best:.6f}"
            if self.verbose:
                print_log(s)
            return True

        prev_best = self.best
        if self.mode == "max":  # larger better
            self.best = max(self.best, criterion_value)
        else:  # smaller better
            self.best = min(self.best, criterion_value)

        is_updated = (self.best == criterion_value)
        if is_updated:
            self._patience_count = 0
            if self.verbose:
                s = f"...... Plateau best updated, "
                s += f"(old/new): ({prev_best:.6f} / {self.best:.6f})"
                print_log(s, force_print=True)
        else:
            self._patience_count += 1
            if self.verbose:
                s = f"...... Plateau best NOT updated, "
                s += f"(old/new): ({prev_best:.6f} / {criterion_value:.6f})\n"
                s += f"......... patience increased ({self._patience_count} / {self.patience})"
                print_log(s, force_print=True)
            if self._patience_count >= self.patience:
                self._patience_count = 0  # should we?
                self._drop_count += 1
                if self.verbose:
                    s = f"......... drop count increased ({self._drop_count} times dropped)"
                    print_log(s, force_print=True)
        return is_updated

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if self.num_iterations < self.warmup_iterations:
            lr = initial_lr * (self.num_iterations + 1) / self.warmup_iterations
        elif self.num_iterations < self.warmup_iterations + self.init_keep_iterations:
            lr = initial_lr
        else:
            lr = initial_lr * math.pow(self.multiply_factor, self._drop_count)
            lr = max(lr, self.min_lr)
        return lr

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        config = cls._scale_config(config, dataloader)
        return cls(
            optimizer=optimizer,
            multiply_factor=config.get("multiply_factor", 0.1),
            patience=config.get("patience", 1),
            warmup_iterations=config.get("warmup_iterations", 0),
            init_keep_iterations=config.get("init_keep_iterations", 0),
            min_lr=config.get("min_lr", 1e-8),
            mode=config.get("mode", "min")
        )
