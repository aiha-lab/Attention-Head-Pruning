from typing import Dict, Any, Callable, List, Optional, Tuple, Union
import copy

import torchvision.transforms as torchVT
import torchaudio.transforms as torchAT
from nnlib.nn import Sequential

__all__ = [
    "build_transforms", "register_transform", "BaseTransform", "IndexCompose",
    "SentencePieceWrapper", "GraphemeWrapper",
]

TRANSFORM_REGISTRY = {}


def register_transform(name: str):
    """Decorator to register transform."""

    def register_transform_cls(cls: Callable):
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"[ERROR:DATA] Cannot register duplicated transform {name}.")
        if hasattr(torchVT, name):
            raise ValueError(f"[ERROR:DATA] Cannot register same transform {name} as TorchVision.")
        if hasattr(torchAT, name):
            raise ValueError(f"[ERROR:DATA] Cannot register same transform {name} as TorchAudio.")
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return register_transform_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom transforms here (do manually for transparency.)
# from .XXX import YYY
from .transform import BaseTransform
from .compose import IndexCompose
from .sentencepiece import SentencePieceWrapper
from .grapheme import GraphemeWrapper


# -------------------------------------------------------------------------------- #


def build_transform(transform_config: Dict[str, Any]) -> Callable:
    """Build single transform.
    transform_config: {"name": "xxx", "kwargs": "xxx"}
    """
    if "name" not in transform_config:
        raise ValueError("[ERROR:DATA] Transform build should have name on it.")
    name = transform_config["name"]
    config = copy.deepcopy(transform_config)
    del config["name"]

    if name in TRANSFORM_REGISTRY:
        t = TRANSFORM_REGISTRY[name].from_config(config)
    elif hasattr(torchVT, name):
        t = getattr(torchVT, name)(**config)
    elif hasattr(torchAT, name):
        t = getattr(torchAT, name)(**config)
    else:
        raise ValueError(f"[ERROR:DATA] Transform {name} is neither in TorchVision, TorchAudio, nor HappyTorch.")
    return t


def build_transforms(transform_configs_before: Optional[List[Dict[str, Any]]],
                     transform_configs_after: Optional[List[Dict[str, Any]]] = None
                     ) -> Union[None, Callable, Tuple[Callable, Callable]]:
    """Build Transform function."""
    if transform_configs_before is not None:
        transform_list_before = [build_transform(c) for c in transform_configs_before]
        transform_before = IndexCompose(transform_list_before)
    else:
        transform_before = None

    if transform_configs_after is not None:
        transform_list_after = [build_transform(c) for c in transform_configs_after]
        transform_after = Sequential(transform_list_after)
    else:
        transform_after = None

    if (transform_before is not None) and (transform_after is not None):
        return transform_before, transform_after
    elif transform_before is not None:
        return transform_before
    elif transform_after is not None:
        return transform_after
    else:
        return None
