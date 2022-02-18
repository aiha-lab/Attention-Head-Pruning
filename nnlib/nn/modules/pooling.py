from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.modules.utils import to_single_tuple, to_pair_tuple, to_triple_tuple


class _MaxPoolNd(BaseModule):

    def __init__(self, kernel_size, stride=None,
                 padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}" \
               f", dilation={self.dilation}, ceil_mode={self.ceil_mode}"


class MaxPool1d(_MaxPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                            self.ceil_mode, self.return_indices)


class MaxPool2d(_MaxPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                            self.ceil_mode, self.return_indices)


class MaxPool3d(_MaxPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                            self.ceil_mode, self.return_indices)


class _AvgPoolNd(BaseModule):

    def __init__(self):
        super(_AvgPoolNd, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool1d(_AvgPoolNd):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode: bool = False, count_include_pad: bool = True):
        super(AvgPool1d, self).__init__()
        self.kernel_size = to_single_tuple(kernel_size)
        self.stride = to_single_tuple(stride if stride is not None else kernel_size)
        self.padding = to_single_tuple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)


class AvgPool2d(_AvgPoolNd):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[bool] = None):
        super(AvgPool2d, self).__init__()
        self.kernel_size = to_pair_tuple(kernel_size)
        self.stride = to_pair_tuple(stride if stride is not None else kernel_size)
        self.padding = to_pair_tuple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad, self.divisor_override)


class AvgPool3d(_AvgPoolNd):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[bool] = None):
        super(AvgPool3d, self).__init__()
        self.kernel_size = to_triple_tuple(kernel_size)
        self.stride = to_triple_tuple(stride if stride is not None else kernel_size)
        self.padding = to_triple_tuple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad, self.divisor_override)


class GlobalAvgPool2d(BaseModule):

    def __init__(self, keepdim: bool = False):
        super(GlobalAvgPool2d, self).__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("[ERROR:NN] GlobalAvgPool2d input should be 4D tensor.")
        return torch.mean(x, dim=[2, 3], keepdim=self.keepdim)


class _AdaptiveMaxPoolNd(BaseModule):

    def __init__(self, output_size, return_indices: bool = False) -> None:
        super(_AdaptiveMaxPoolNd, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}" + (
            f", return_indices={self.return_indices}" if self.return_indices else "")


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return F.adaptive_max_pool1d(x, self.output_size, self.return_indices)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return F.adaptive_max_pool2d(x, self.output_size, self.return_indices)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return F.adaptive_max_pool3d(x, self.output_size, self.return_indices)


class _AdaptiveAvgPoolNd(BaseModule):

    def __init__(self, output_size) -> None:
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)


class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, self.output_size)
