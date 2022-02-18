from typing import Dict, Union, Any
import torch

from happy_torch.data.transforms import register_transform
from happy_torch.data.transforms.transform import BaseTransform


@register_transform("WordAugment")
class WordAugment(BaseTransform):

    def __init__(self,
                 vocab_size: int,
                 num_swap: Union[int, float] = 0,
                 num_replace: Union[int, float] = 0):
        super(WordAugment, self).__init__()
        self.vocab_size = vocab_size

        if isinstance(num_swap, float):
            if not (0 <= num_swap < 1):
                raise ValueError(f"[ERROR:DATA] WordAug num_swap should be in range [0, 1), but got {num_swap}.")
        if isinstance(num_replace, float):
            if not (0 <= num_replace < 1):
                raise ValueError(f"[ERROR:DATA] WordAug num_replace should be in range [0, 1), but got {num_replace}.")

        self.num_swap = num_swap
        self.num_replace = num_replace

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        """
        s:      (batch_size, num_sequence)
        """
        orig_shape = s.shape
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if s.ndim != 2:
            raise ValueError(f"[ERROR:DATA] WordAugment require input to be 1D/2D, but got {s.shape}.")

        batch_size, seq_len = s.shape
        if self.num_swap > 0:
            if isinstance(self.num_swap, float):
                num_swap = int(self.num_swap * seq_len)
            else:
                num_swap = self.num_swap

            for _ in range(num_swap):  # shuffle order: [0, 1, 2, 3] -> [0, 2, 1, 3]
                pos = torch.randint(0, seq_len - 1, (1,)).item()
                s_left = s[:, pos].clone()
                s[:, pos] = s[:, pos + 1]
                s[:, pos + 1] = s_left

        if self.num_replace > 0:
            if isinstance(self.num_replace, float):
                num_replace = int(self.num_replace * seq_len)
            else:
                num_replace = self.num_replace

            for _ in range(num_replace):  # random replace: [0, 1, 2, 3] -> [0, 1, 31, 3]
                pos = torch.randint(0, seq_len, (1,)).item()
                rep = torch.randint(0, self.vocab_size, (batch_size,), dtype=s.dtype, device=s.device)
                s[:, pos] = rep

        s = s.view(*orig_shape)
        return s.contiguous()

    @classmethod
    def from_config(cls, config: Dict[str, Any], vocab_size: int):
        return cls(
            vocab_size=vocab_size,
            num_swap=config.get("num_swap", 0),
            num_replace=config.get("num_replace", 0)
        )
