from typing import Any, List, Dict
import numpy as np
import time
import math
import copy

from nnlib.utils.print_utils import print_log
from nnlib.utils.dist_utils import is_master, broadcast_objects, is_distributed


class AutoRegressiveLMWrapper(object):

    def __init__(self,
                 dataset,
                 batch_size: int,
                 seq_length: int,
                 overlap_length: int = 0,
                 shuffle: bool = False,
                 shard_idx: int = 0,
                 num_shards: int = 1):
        # ---------------------------------------------------------------- #
        try:
            _ = dataset.data
        except AttributeError:
            raise AttributeError(f"[ERROR:DATA] Dataset should have .data attribute.")

        self.dataset = dataset
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.overlap_length = overlap_length

        if not (0 <= shard_idx < num_shards):
            raise ValueError(f"[ERROR:DATA] Shard index {shard_idx} not in range. Total shards: {num_shards}.")
        self.shard_idx = shard_idx
        self.num_shards = num_shards

        self.batch_indices = []
        self.build()
        if len(self) == 0:
            raise ValueError(f"[ERROR:DATA] Batch indices empty, not correctly set for rank {shard_idx}.")
        if len(self) % num_shards != 0:
            raise ValueError(f"[ERROR:DATA] num_batches {len(self)} is not multiple of num_shards {num_shards}.")

    def build(self) -> None:
        """Build batched dataset. batching is only done at master, then broadcast to others."""
        if (not self.shuffle) and (len(self) > 0):  # built before
            print_log("[WARN:DATA] Shuffle is False, but you called build again. Is this by purpose?")

        if is_master():
            start_time = time.process_time()
            print_log(f"[LOG:DATA] AutoRegressiveLMWrapper {self.dataset.mode} build called, shuffle={self.shuffle}.")

            # keep data slice
            # DistributedSampler will distribute index to:
            # [0, 4, 8,  12, ...] -> rank0
            # [1, 5, 9,  13, ...] -> rank1
            # [2, 6, 10, 14, ...] -> rank2
            # [3, 7, 11, 15, ...] -> rank3

            data_per_shard = len(self.dataset) // self.num_shards

    def __len__(self) -> int:
        return len(self.batch_indices)
