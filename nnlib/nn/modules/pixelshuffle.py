import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule


class PixelShuffle(BaseModule):

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x, self.upscale_factor)

    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"
