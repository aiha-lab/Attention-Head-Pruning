from typing import Dict, Any, List
import copy

import torch.nn.modules.loss as torchLoss

__all__ = [
    "build_losses", "register_loss",
    "CELossWithSmoothing",
    "L1LossToZero",
    "L2LossToZero",
]

LOSS_REGISTRY = {}


def register_loss(name: str):
    """Decorator to register loss."""

    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"[ERROR:LOSS] Cannot register duplicated loss {name}.")
        if hasattr(torchLoss, name):
            raise ValueError(f"[ERROR:LOSS] Cannot register same loss {name} as pytorch.")
        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom loss here (do manually for transparency.)
# from .XXX import YYY
from .cross_entropy import CELossWithSmoothing
from .norm import L1LossToZero, L2LossToZero


# -------------------------------------------------------------------------------- #

def build_loss(loss_config: Dict[str, Any]):
    """Build a single loss."""
    if "name" not in loss_config:
        raise ValueError("[ERROR:LOSS] Loss build should have name on it.")
    name = loss_config["name"]
    config = copy.deepcopy(loss_config)
    del config["name"]
    if "coefficient" in config:
        del config["coefficient"]

    if name in LOSS_REGISTRY:
        loss = LOSS_REGISTRY[name].from_config(config)
    elif hasattr(torchLoss, name):
        loss = getattr(torchLoss, name)(**config)
    else:
        raise ValueError(f"[ERROR:LOSS] Loss {name} is neither in pytorch nor HappyTorch.")
    return loss


def build_losses(loss_configs: List[Dict[str, Any]]):
    losses = [build_loss(c) for c in loss_configs]
    coefficients = []
    for c in loss_configs:
        cf = c["coefficient"] if ("coefficient" in c) else 1.0
        coefficients.append(cf)
    return losses, coefficients
