from typing import Dict, Any, List
import copy

__all__ = [
    "build_metrics", "register_metric",
    "Accuracy",
]

METRIC_REGISTRY = {}


def register_metric(name: str):
    """Decorator to register metric."""

    def register_metric_cls(cls):
        if name in METRIC_REGISTRY:
            raise ValueError(f"[ERROR:METRIC] Cannot register duplicated metric {name}.")
        METRIC_REGISTRY[name] = cls
        return cls

    return register_metric_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom metric here (do manually for transparency.)
# from .XXX import YYY
from .accuracy import Accuracy


# -------------------------------------------------------------------------------- #

def build_metric(metric_config: Dict[str, Any]):
    """Build single metric."""
    if "name" not in metric_config:
        raise ValueError("[ERROR:METRIC] Metric build should have name on it.")
    name = metric_config["name"]
    config = copy.deepcopy(metric_config)
    del config["name"]
    if name in METRIC_REGISTRY:
        metric = METRIC_REGISTRY[name].from_config(config)
    else:
        raise ValueError(f"[ERROR:METRIC] Metric {name} is not in HappyTorch.")
    return metric


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [build_metric(c) for c in metric_configs]
    return metrics
