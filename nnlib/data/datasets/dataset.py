from typing import Any, Dict, List
import bisect
import numpy as np


class HappyDataset(object):

    def __init__(self, transform=None, target_transform=None, **kwargs):
        self.transform = transform
        self.target_transform = target_transform

    def set_transform(self, transform):
        self.transform = transform

    def set_target_transform(self, transform):
        self.target_transform = transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any], transform=None, target_transform=None):
        raise NotImplementedError


class HappyConcatDataset(HappyDataset):

    def __init__(self, datasets: List[HappyDataset]) -> None:
        if not isinstance(datasets, list):
            raise ValueError(f"[ERROR:DATA] ConcatDataset requires List input, but got {type(datasets)}.")
        super(HappyConcatDataset, self).__init__()
        self.datasets = datasets

        for d in self.datasets:
            if len(d) == 0:
                raise ValueError(f"[ERROR:DATA] ConcatDataset {d.__class__.__name__} is empty.")
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.cumsum = np.cumsum(self.dataset_lengths).tolist()

    @property
    def num(self):
        return len(self.datasets)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            if -index > len(self):
                raise IndexError("[ERROR:DATA] Index length overflow, exceed dataset length.")
            index = len(self) + index
        elif index >= len(self):
            raise IndexError("[ERROR:DATA] Index length overflow, exceed dataset length.")

        dataset_index = bisect.bisect_right(self.cumsum, index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumsum[dataset_index - 1]
        return self.datasets[dataset_index][sample_index]

    @classmethod
    def from_config(cls, config: Dict[str, Any], transform=None, target_transform=None):
        raise RuntimeError("[ERROR:DATA] HappyConcatDataset cannot be initialized from config.")
