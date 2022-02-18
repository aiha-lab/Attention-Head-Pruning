from typing import Dict, Any, List
import copy

import torchvision.datasets as torchVisionData
import torchaudio.datasets as torchAudioData  # NOT YET

# import torchtext.datasets as torchTextData  # NOT YET must-install nnlib.

__all__ = [
    "build_dataset", "build_datasets", "register_dataset", "HappyDataset", "HappyConcatDataset",
    "HappyWikitext2",
    "HappyWikitext103",
    "HappyText8",
    "HappySentences",
]

DATASET_REGISTRY = {}


def register_dataset(name: str):
    """Decorator to register dataset."""

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"[ERROR:DATA] Cannot register duplicated dataset {name}.")
        if hasattr(torchVisionData, name):
            raise ValueError(f"[ERROR:DATA] Cannot register same dataset {name} as TorchVision.")
        if hasattr(torchAudioData, name):
            raise ValueError(f"[ERROR:DATA] Cannot register same dataset {name} as TorchAudio.")
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


# -------------------------------------------------------------------------------- #
# IMPORT custom datasets here (do manually for transparency.)
# from .XXX import YYY
from .dataset import HappyConcatDataset, HappyDataset
from .wikitext import HappyWikitext2, HappyWikitext103
from .text8 import HappyText8
from .sentences import HappySentences


# -------------------------------------------------------------------------------- #

def build_dataset(dataset_config: Dict[str, Any], transform=None, target_transform=None):
    """Build dataset."""
    if "name" not in dataset_config:
        raise ValueError("[ERROR:DATA] Dataset build should have name on it.")
    name = dataset_config["name"]
    config = copy.deepcopy(dataset_config)
    del config["name"]

    if name in DATASET_REGISTRY:
        d_set = DATASET_REGISTRY[name].from_config(config, transform=transform, target_transform=target_transform)
    elif hasattr(torchVisionData, name):
        d_set = getattr(torchVisionData, name)(**config, transform=transform, target_transform=target_transform)
    elif hasattr(torchAudioData, name):
        d_set = getattr(torchAudioData, name)(**config)  # no transform nor target_transform.
    else:
        raise ValueError(f"[ERROR:DATA] Dataset {name} is not in TorchVision, TorchAudio nor HappyTorch.")
    return d_set


def build_datasets(dataset_configs: List[Dict[str, Any]], transform=None, target_transform=None):
    """Build multiple datasets"""
    datasets = [build_dataset(c, transform=transform, target_transform=target_transform)
                for c in dataset_configs]
    datasets = HappyConcatDataset(datasets)
    return datasets
