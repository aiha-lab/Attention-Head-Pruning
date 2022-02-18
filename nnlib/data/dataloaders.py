from typing import Dict, Any, Callable
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nnlib.utils.dist_utils import is_distributed, get_world_size


class HappyDataLoader(DataLoader):

    def __init__(self,
                 dataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0, *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 sampler=None,
                 collate_fn=None) -> None:

        if dist.is_initialized() and (sampler is None):
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
            shuffle = None

        super(HappyDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            sampler=sampler, collate_fn=collate_fn
        )

    def set_collate_fn(self, fn: Callable):
        self.collate_fn = fn
        return self

    def set_epoch(self, epoch: int):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
        return self

    @classmethod
    def from_config(cls, config: Dict[str, Any], dataset,
                    sampler=None, collate_fn=None) -> "HappyDataLoader":
        return cls(
            dataset=dataset,
            batch_size=config.get("batch_size"),
            shuffle=config.get("shuffle", False),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
            drop_last=config.get("drop_last", False),
            timeout=config.get("timeout", 0.0),
            prefetch_factor=config.get("prefetch_factor", 2),
            persistent_workers=config.get("persistent_workers", False),
            sampler=sampler,
            collate_fn=collate_fn,
        )


class WrapperDataLoader(DataLoader):
    """Wrapper dataloader.
    For super(), batch_size=1, shuffle=False. drop_last=False
    Shuffle should be handled by wrapper.
    len(dataset) is recommended to be a multiple of world_size.
    """

    def __init__(self,
                 dataset,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 timeout: float = 0, *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False) -> None:

        if is_distributed():
            if len(dataset) % get_world_size() != 0:
                raise ValueError(f"[ERROR:DATALOADER] Wrapper dataset length is not multiple of world_size.")
            sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
            shuffle = None
        else:
            sampler = None
            shuffle = False

        super(WrapperDataLoader, self).__init__(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            sampler=sampler, collate_fn=self._naive_collate
        )

    def set_epoch(self, epoch: int):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
        return self

    def reset_sampler(self) -> None:
        if isinstance(self.sampler, DistributedSampler):
            if len(self.dataset) % self.sampler.num_replicas != 0:
                raise ValueError(f"[ERROR:DATALOADER] Wrapper dataset length is not multiple of world size.")
            # ad-hoc fix
            self.sampler.num_samples = len(self.dataset) // self.sampler.num_replicas
            self.sampler.total_size = len(self.dataset)

    @classmethod
    def from_config(cls, config: Dict[str, Any], dataset) -> "WrapperDataLoader":
        return cls(
            dataset=dataset,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
            timeout=config.get("timeout", 0.0),
            prefetch_factor=config.get("prefetch_factor", 2),
            persistent_workers=config.get("persistent_workers", False),
        )

    @staticmethod
    def _naive_collate(batch):
        # return first one because always batch_size is 1.
        return batch[0]
