from typing import Tuple, Dict, Any
import math
import copy
import torch
import numpy as np


class AutoRegressiveLMIterator(object):
    """Wrapper of AR-LM dataset.
    Each GPU see equal split of data (fixed).
    """

    def __init__(self,
                 dataset,
                 batch_size: int,
                 seq_length: int,
                 overlap_length: int = 0,
                 shard_idx: int = 0,
                 num_shards: int = 1):
        # ---------------------------------------------------------------- #
        try:
            data = dataset.data
        except AttributeError:
            raise AttributeError(f"[ERROR:LOADER] Dataset should have .data attribute.")

        if overlap_length >= seq_length:
            raise ValueError(f"[ERROR:LOADER] Overlap length {overlap_length} >= sequence length {seq_length}.")

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.overlap_length = overlap_length

        if not (0 <= shard_idx < num_shards):
            raise ValueError(f"[ERROR:LOADER] Shard index {shard_idx} not in range. Total shards: {num_shards}.")
        self.shard_idx = shard_idx
        self.num_shards = num_shards

        data_per_shard = len(data) // num_shards
        start_idx = shard_idx * data_per_shard
        end_idx = (shard_idx + 1) * data_per_shard

        raw_data = np.array(copy.deepcopy(data[start_idx:end_idx]), dtype=np.int64)

        self.num_steps = len(raw_data) // batch_size
        # trim off dataset
        raw_data = raw_data[:self.num_steps * batch_size]
        raw_data = np.reshape(raw_data, (self.batch_size, self.num_steps))  # (batch_size, num_steps)
        self.data = raw_data  # (batch_size, num_steps)

        # self.num_batch = (self.num_steps + self.seq_length - 1 - self.overlap_length) // (
        #         self.seq_length - self.overlap_length)
        self.num_batch = int(math.ceil((self.num_steps - 1 - self.seq_length) /
                                       (self.seq_length - self.overlap_length))) + 1

    def __len__(self):
        return self.num_batch

    def set_collate_fn(self, fn=None):
        pass  # just for consistency

    def set_epoch(self, epoch: int):
        pass  # just for consistency

    def get_batch(self, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        end_idx = min(start_idx + self.seq_length, self.num_steps - 1)  # exclusive
        seq_len = end_idx - start_idx

        data = self.data[:, start_idx:end_idx]
        label = self.data[:, start_idx + 1:end_idx + 1]

        data_t = torch.as_tensor(data, dtype=torch.long)
        label_t = torch.as_tensor(label, dtype=torch.long)

        return data_t, label_t, seq_len

    def shuffle(self):
        random_idx = np.random.randint(1, self.num_steps - 1, (self.batch_size,))
        for i in range(self.batch_size):
            shift = random_idx[i]
            row = self.data[i]  # (num_steps,)
            self.data[i] = np.roll(row, shift=shift)

    def get_iterator(self):
        for i in range(0, self.num_steps - 1, self.seq_length - self.overlap_length):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_iterator()

    @classmethod
    def from_config(cls, config: Dict[str, Any], dataset, shard_idx: int = 0, num_shards: int = 1):
        return cls(
            dataset,
            batch_size=config.get("batch_size"),
            seq_length=config.get("seq_length"),
            overlap_length=config.get("overlap_length", 0),
            shard_idx=shard_idx,
            num_shards=num_shards,
        )
