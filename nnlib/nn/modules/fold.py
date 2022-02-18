import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule


class Fold(BaseModule):

    def __init__(self,
                 output_size,
                 kernel_size,
                 dilation=(1, 1),
                 padding=(0, 0),
                 stride=(1, 1)) -> None:
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.fold(x,
                      self.output_size,
                      self.kernel_size,
                      dilation=self.dilation,
                      padding=self.padding,
                      stride=self.stride)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, kernel_size={self.kernel_size}, " \
               f"dilation={self.dilation}, padding={self.padding}, stride={self.stride}"


class Unfold(BaseModule):

    def __init__(self,
                 kernel_size,
                 dilation=(1, 1),
                 padding=(0, 0),
                 stride=(1, 1)) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.unfold(x,
                        self.kernel_size,
                        dilation=self.dilation,
                        padding=self.padding,
                        stride=self.stride)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, dilation={self.dilation}, padding={self.padding}, " \
               f"stride={self.stride}"
