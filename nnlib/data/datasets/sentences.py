from typing import Optional, Any, Dict, Tuple
import os
import torch

from nnlib.data.datasets import register_dataset
from nnlib.data.datasets.dataset import HappyDataset
from nnlib.nn.utils import pad_sequence

from nnlib.data.helpers.utils import english_cleaners


@register_dataset("HappySentences")
class HappySentences(HappyDataset):
    """Common dataset which use SentencePiece and txt file."""

    def __init__(self,
                 file_path: str,
                 transform: Optional = None,
                 clean_script: bool = True):
        if transform is None:
            raise ValueError("[ERROR:DATA] Sentences dataset require transform.")

        super(HappySentences, self).__init__(transform=transform)
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise ValueError(f"[ERROR:DATA] Sentences dataset path {self.file_path} does not exist.")

        self.clean_script = clean_script

        # keep all text to RAM after preprocess.
        data = []
        with open(self.file_path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                if self.clean_script:
                    line = english_cleaners(line)
                indices = self.transform(line)
                data.append(indices)
        self.dataset = data

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:

        indices = self.dataset[index]  # Tensor
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        return indices

    @classmethod
    def from_config(cls, config: Dict[str, Any], transform=None, target_transform=None):
        return cls(
            file_path=config.get("file_path"),
            transform=transform,
            clean_script=config.get("clean_script", True),
        )

    @staticmethod
    def fast_collate_sentences(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch includes [indices]
        Return
            indices:        [batch_size, max_length]    long
            length:         [batch_size]    long
        """
        length = torch.tensor([b.shape[0] for b in batch], dtype=torch.long)
        indices = pad_sequence(batch, output_batch_first=True, padding_value=0, pad_to_multiple=1)
        return indices, length
