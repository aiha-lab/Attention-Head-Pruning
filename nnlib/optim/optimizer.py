from typing import Tuple, Dict, Any, Union, List, Optional
from torch.optim.optimizer import Optimizer

from nnlib.optim.lr_scheduler.scheduler import BaseScheduler


class BaseOptimizer(Optimizer):

    def __init__(self, params, defaults):
        super(BaseOptimizer, self).__init__(params, defaults)
        self.scheduler: Optional[BaseScheduler] = None  # will be set by scheduler

    def current_lrs(self) -> Tuple[float, ...]:
        lrs = []
        for param_group in self.param_groups:
            lrs.append(param_group["lr"])
        assert len(lrs) > 0
        return tuple(lrs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], params):
        raise NotImplementedError

    def force_lr(self, lr: float):
        """Force set learning rate. To change LR after state load."""
        # currently only support same LRs for all param_groups.
        for pg in self.param_groups:
            pg["lr"] = lr
            if hasattr(pg, "initial_lr") or (self.scheduler is not None):
                pg["initial_lr"] = lr

        if self.scheduler is not None:
            self.scheduler.base_lrs = list(map(lambda g: g["initial_lr"], self.param_groups))


class OptimizerList(object):

    def __init__(self, optimizers: Union[BaseOptimizer, List[BaseOptimizer]]):
        if isinstance(optimizers, BaseOptimizer):
            optimizers = [optimizers]
        self.optimizers = optimizers

    def __len__(self) -> int:
        return len(self.optimizers)

    def current_lrs(self) -> List[Tuple[float, ...]]:
        lrs = []
        for opt in self.optimizers:
            lrs.append(opt.current_lrs())
        return list(lrs)

    def step(self, closure=None) -> None:
        for opt in self.optimizers:
            opt.step(closure=closure)

    def state_dict(self) -> List:
        states = []
        for opt in self.optimizers:
            states.append(opt.state_dict())
        return states

    def load_state_dict(self, state_dict: List) -> None:
        # assume state_dict will loaded as same order as saved.
        if len(state_dict) != len(self):
            raise ValueError(f"[ERROR:OPT] Load state dict #optimizers mismatch, {len(self)} vs {len(state_dict)}.")
        for state, opt in zip(state_dict, self.optimizers):
            opt.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def force_lr(self, lr: Union[float, List[float]]):
        # currently only supports setting LR all same.
        if isinstance(lr, float):
            lr = [lr] * len(self)
        for r, opt in zip(lr, self.optimizers):
            opt.force_lr(r)
