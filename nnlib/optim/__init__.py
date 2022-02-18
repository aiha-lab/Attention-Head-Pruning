from typing import Dict, Any
import copy

import torch.optim as torchOpt

__all__ = [
    "build_optimizer", "register_optimizer", "OptimizerList",
    "HappySGD",
    "HappyAdam",
    "HappyLamb",
    "HappyRMSProp",
    "HappyRAdam",
    "HappyAdamP",
    "HappySGDP",
]

OPTIMIZER_REGISTRY = {}


def register_optimizer(name: str):
    """Decorator to register optimizer."""

    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError(f"[ERROR:OPT] Cannot register duplicated optimizer {name}.")
        if hasattr(torchOpt, name):
            raise ValueError(f"[ERROR:OPT] Cannot register same optimizer {name} as pytorch.")
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return register_optimizer_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom optimizers here (do manually for transparency.)
# from .XXX import YYY
from .optimizer import BaseOptimizer, OptimizerList
from .sgd import HappySGD
from .rmsprop import HappyRMSProp
from .adam import HappyAdam
from .lamb import HappyLamb
from .radam import HappyRAdam
from .adamp import HappyAdamP
from .sgdp import HappySGDP


# -------------------------------------------------------------------------------- #

def build_optimizer(optimizer_config: Dict[str, Any], params) -> BaseOptimizer:
    """Build optimizer."""
    if "name" not in optimizer_config:
        raise ValueError("[ERROR:OPT] Optimizer build should have name on it.")
    name = optimizer_config["name"]
    config = copy.deepcopy(optimizer_config)
    del config["name"]

    if name in OPTIMIZER_REGISTRY:
        opt = OPTIMIZER_REGISTRY[name].from_config(config, params)
    elif hasattr(torchOpt, name):
        opt = getattr(torchOpt, name)(params, **config)
    else:
        raise ValueError(f"[ERROR:OPT] Optimizer {name} is not in pytorch nor HappyTorch.")
    return opt
