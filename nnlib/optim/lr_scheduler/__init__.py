from typing import Dict, Any
import copy

import torch.optim.lr_scheduler as torchLR

__all__ = [
    "build_scheduler", "register_scheduler", "SchedulerList",
    "HappyCosineLR",
    "HappyStepLR",
    "HappyExponentialLR",
    "HappyReduceLROnPlateau",
    "HappyInvSqrtLR",
    "HappyConstantLR",
]

SCHEDULER_REGISTRY = {}


def register_scheduler(name: str):
    """Decorator to register scheduler."""

    def register_scheduler_cls(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"[ERROR:SCHED] Cannot register duplicated optimizer {name}.")
        if hasattr(torchLR, name):
            raise ValueError(f"[ERROR:SCHED] Cannot register same optimizer {name} as pytorch.")
        SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_scheduler_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom schedulers here (do manually for transparency.)
# from .XXX import YYY
from .scheduler import BaseScheduler, SchedulerList
from .cosine import HappyCosineLR
from .exponential import HappyExponentialLR
from .step import HappyStepLR
from .reduce_plateau import HappyReduceLROnPlateau
from .constant import HappyConstantLR
from .inv_sqrt import HappyInvSqrtLR


# -------------------------------------------------------------------------------- #

def build_scheduler(scheduler_config: Dict[str, Any], optimizer, dataloader=None) -> BaseScheduler:
    """Build scheduler."""
    if "name" not in scheduler_config:
        raise ValueError("[ERROR:SCHED] Scheduler build should have name on it.")
    name = scheduler_config["name"]
    config = copy.deepcopy(scheduler_config)
    del config["name"]

    if name in SCHEDULER_REGISTRY:
        sched = SCHEDULER_REGISTRY[name].from_config(config, optimizer, dataloader=dataloader)
    elif hasattr(torchLR, name):
        sched = getattr(torchLR, name)(optimizer, **config)
    else:
        raise ValueError(f"[ERROR:SCHED] Scheduler {name} is not in pytorch nor HappyTorch.")
    return sched
