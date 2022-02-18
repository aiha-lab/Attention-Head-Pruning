from typing import Optional, Union
import torch

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.functional import gradient_scale


class TimeGradientScale(BaseModule):

    def __init__(self, dim: int = 1):
        super(TimeGradientScale, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, length: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if length is None:
            seq_length = x.shape[self.dim]
            scale = torch.ones(batch_size, dtype=x.dtype, device=x.device).div_(seq_length)
        else:
            if isinstance(length, int):
                length = torch.ones(batch_size, dtype=torch.long, device=x.device).mul_(length)
            scale = 1.0 / length.float()

        if scale.shape != (batch_size,):
            raise ValueError(f"[ERROR:NN] TimeGradientScale needs length to be same as batch.")

        scale_shape = (batch_size,) + tuple(1 for _ in range(x.ndim - 1))
        scale = scale.view(*scale_shape)

        return gradient_scale(x, scale)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
