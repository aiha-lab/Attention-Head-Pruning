from typing import Any, Dict
import os
import torch

from nnlib.data.datasets import register_dataset
from nnlib.data.datasets.dataset import HappyDataset
from nnlib.data.helpers.dictionary import Dictionary, ENGLISH_CHARACTERS
from nnlib.utils.print_utils import print_log
from nnlib.utils.dist_utils import is_master


@register_dataset("HappyText8")
class HappyText8(HappyDataset):
    """Text8 for character-level language modeling"""

    def __init__(self,
                 data_dir: str,
                 mode: str = "train"):
        super(HappyText8, self).__init__()
        # ---------------------------------------------------------------- #
        mode = mode.lower()
        if mode not in ("train", "valid", "test"):
            raise ValueError(f"[ERROR:DATA] Text8 dataset mode {mode} is unsupported.")

        self.mode = mode
        data_path = os.path.join(data_dir, "text8")

        self.dictionary = Dictionary()
        for c in ENGLISH_CHARACTERS:
            self.dictionary.add_token(c, n=0)

        self.data = []

        cached_data_path = os.path.join(data_dir, f"{mode}.cache.pth")
        if os.path.isfile(cached_data_path):
            self.data = torch.load(cached_data_path)
        else:
            print(f"[LOG:DATA] Creating cache data for Text8 {mode}.")
            with open(data_path, "r", encoding="utf-8") as f:
                line = f.read()  # only 1 line is in dataset
                assert len(line) == 100 * 1000 * 1000  # 100M
                if mode == "train":
                    raw = line[:90 * 1000 * 1000]  # 90M
                elif mode == "valid":
                    raw = line[90 * 1000 * 1000:95 * 1000 * 1000]  # 5M
                else:  # test
                    raw = line[95 * 1000 * 1000:]  # 5M

                raw_p = ["_" if c == " " else c for c in raw]
                for c in raw_p:
                    char_idx = self.dictionary.get_token_idx(c, use_unknown=False)
                    self.data.append(char_idx)

            if is_master():
                torch.save(self.data, cached_data_path)

        assert len(self.dictionary) == 27

        s = f"[LOG:LM] text8 {self.mode}: {len(self.data)} words, {len(self.dictionary)} vocabulary."
        print_log(s)

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        # should not be used.
        return self.data[index]

    @classmethod
    def from_config(cls, config: Dict[str, Any], transform=None, target_transform=None):
        return cls(
            data_dir=config.get("data_dir"),
            mode=config.get("mode", "train"),
        )
