from typing import Optional
import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule


class Upsample(BaseModule):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode: str = "nearest",
                 align_corners: Optional[bool] = None) -> None:
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

        if (self.scale_factor is None) and (self.size is None):
            raise ValueError("[ERROR:NN] Upsample size or scale_factor should not be both None.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            s = f"scale_factor={self.scale_factor}"
        else:
            s = f"size={self.size}"
        s += f", mode={self.mode}"
        return s


class UpsamplingNearest2d(Upsample):

    def __init__(self,
                 size=None,
                 scale_factor=None) -> None:
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 align_corners: bool = True) -> None:
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor, mode="bilinear", align_corners=align_corners)
