from typing import Optional, Dict, Any, List, Union
import copy
import torch

from nnlib.utils.print_utils import print_log


class BaseScheduler(object):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_iterations: int = 0,
                 init_keep_iterations: int = 0,
                 min_lr: float = 1e-8,
                 mode: str = "min",
                 *, verbose: bool = True) -> None:

        self.optimizer = optimizer
        self.optimizer.scheduler = self
        self.best = None

        self.mode = mode.lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"[ERROR:SCHED] Scheduler mode should be either min or max, got {self.mode}.")

        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group.setdefault("initial_lr", param_group["lr"])
                # raise KeyError("[ERROR:SCHED] Attribute `initial_lr` is not set. Maybe using different optimizer?")

        self.base_lrs = list(map(lambda g: g["initial_lr"], optimizer.param_groups))
        self.num_iterations = -1
        self.warmup_iterations = warmup_iterations
        self.init_keep_iterations = init_keep_iterations
        self.min_lr = min_lr
        self.verbose = verbose

        self._patience_count = 0
        self._step_called = False  # simple flag to check if step() is called at least once

    def state_dict(self) -> dict:
        return {
            "best": self.best,  # float
            "num_iterations": self.num_iterations,  # int
            "warmup_iterations": self.warmup_iterations,  # int
            "init_keep_iterations": self.init_keep_iterations,  # int
            "mode": self.mode,  # str
            "min_lr": self.min_lr,  # float
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.best = state_dict.get("best", None)
        self.num_iterations = state_dict.get("num_iterations", -1)
        self.warmup_iterations = state_dict.get("warmup_iterations", 0)
        self.init_keep_iterations = state_dict.get("init_keep_iterations", 0)
        self.mode = state_dict.get("mode", "min")
        self.min_lr = state_dict.get("min_lr", 1e-8)
        self._step_called = True

    def update_best(self, criterion_value) -> bool:
        """Update best validation criterion and return if the value is updated."""
        if self.best is None:
            self.best = criterion_value
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
                s = f"...... best updated, "
                s += f"(old/new): ({prev_best:.6f} / {self.best:.6f})"
                print_log(s)
        else:
            self._patience_count += 1
            if self.verbose:
                s = f"...... best NOT updated, (old/new): ({prev_best:.6f} / {criterion_value:.6f})\n"
                s += f"...... best before: {self._patience_count} checks."
                print_log(s)
        return is_updated

    def step(self, criterion=None, num_iterations: Optional[int] = None) -> None:
        if num_iterations is None:
            self.num_iterations += 1
        else:
            self.num_iterations = num_iterations

        if criterion is not None:
            _ = self.update_best(criterion)

        for i, param_group in enumerate(self.optimizer.param_groups):
            group_lr = self.get_lr(param_group["initial_lr"], param_group_index=i)
            param_group["lr"] = group_lr

        self._step_called = True

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    def _scale_config(config: Dict[str, Any], dataloader=None):
        if dataloader is None:  # cannot apply scaling.
            return config

        config = copy.deepcopy(config)
        to_remove = []
        to_add = []
        for k, v in config.items():
            k = k.lower()
            if "epochs" in k:
                k_iter = k.replace("epochs", "iterations")
                if k_iter in config.keys():
                    raise ValueError(f"[ERROR:SCHED] Both {k} and {k_iter} provided in config.")
                v_iter = int(v * len(dataloader))
                to_remove.append(k)
                to_add.append((k_iter, v_iter))

        for k in to_remove:
            del config[k]
        for k, v in to_add:
            config[k] = v
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer, dataloader=None):
        # don't forget to call _scale_config first.
        raise NotImplementedError


class SchedulerList(object):

    def __init__(self, schedulers: Union[BaseScheduler, List[BaseScheduler]]):
        if isinstance(schedulers, BaseScheduler):
            schedulers = [schedulers]
        self.schedulers = schedulers

    def __len__(self) -> int:
        return len(self.schedulers)

    def update_best(self, criterion_value) -> List[bool]:
        is_updated = []
        for sched in self.schedulers:
            u = sched.update_best(criterion_value=criterion_value)
            is_updated.append(u)
        return is_updated

    def step(self, criterion=None, num_iterations: Optional[int] = None) -> None:
        for sched in self.schedulers:
            sched.step(criterion=criterion, num_iterations=num_iterations)

    def state_dict(self) -> List:
        states = []
        for sched in self.schedulers:
            states.append(sched.state_dict())
        return states

    def load_state_dict(self, state_dict: List) -> None:
        # assume state_dict will loaded as same order as saved.
        if len(state_dict) != len(self):
            raise ValueError(f"[ERROR:OPT] Load state dict #schedulers mismatch, {len(self)} vs {len(state_dict)}.")
        for state, sched in zip(state_dict, self.schedulers):
            sched.load_state_dict(state)

    def __setattr__(self, name, value):
        if name == 'verbose':
            for sched in self.schedulers:
                sched.verbose = value
        super(SchedulerList, self).__setattr__(name, value)
