from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.parameter import ParameterModule
from nnlib.nn.modules.utils import to_single_tuple, to_pair_tuple, to_triple_tuple


class _ConvNd(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed: bool,
                 output_padding,
                 groups,
                 bias: Optional[bool],
                 padding_mode: str) -> None:
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError(f"[ERROR:NN] ConvNd in_channels {in_channels} % groups {groups} != 0.")
        if out_channels % groups != 0:
            raise ValueError(f"[ERROR:NN] ConvNd out_channels {out_channels} % groups {groups} != 0")
        if padding_mode not in ["zeros", "reflect", "replicate", "circular"]:
            raise ValueError(f"[ERROR:NN] ConvNd invalid padding mode {padding_mode}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        # (a, b) -> (b, b, a, a)
        self.reversed_padding_x2 = tuple([x for x in reversed(self.padding) for _ in range(2)])
        if transposed:
            self.weight = ParameterModule(torch.empty(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = ParameterModule(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = ParameterModule(torch.empty(out_channels, ))
        else:
            self.bias = None
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)
        # if self.bias is not None:
        #     oc, ic, kh, kw = self.weight.shape
        #     bound = 1.0 / math.sqrt(ic * kh * kw)  # maybe we should consider groups?
        #     nn.init.uniform_(self.bias.data, -bound, bound)

    def conv_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias() if (self.bias is not None) else None
        return self.conv_core(x, self.weight(), bias)

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}" \
            f", stride={self.stride}"
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (0,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += f", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        return s


class Conv1d(_ConvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: Optional[bool] = True,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_single_tuple(kernel_size)
        stride = to_single_tuple(stride)
        padding = to_single_tuple(padding)
        dilation = to_single_tuple(dilation)

        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                     False, (0,), groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv1d
        self.conv_fn = conv_fn

    def conv_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self.reversed_padding_x2, mode=self.padding_mode)
            return self.conv_fn(x, weight, bias, self.stride,
                                (0,), self.dilation, self.groups)
        return self.conv_fn(x, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)


class Conv2d(_ConvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: Optional[bool] = True,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_pair_tuple(kernel_size)
        stride = to_pair_tuple(stride)
        padding = to_pair_tuple(padding)
        dilation = to_pair_tuple(dilation)

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                     False, (0, 0), groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv2d
        self.conv_fn = conv_fn

    def conv_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self.reversed_padding_x2, mode=self.padding_mode)
            return self.conv_fn(x, weight, bias, self.stride,
                                (0, 0), self.dilation, self.groups)
        return self.conv_fn(x, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)


class Conv3d(_ConvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: Optional[bool] = True,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_triple_tuple(kernel_size)
        stride = to_triple_tuple(stride)
        padding = to_triple_tuple(padding)
        dilation = to_triple_tuple(dilation)

        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                     False, (0, 0, 0), groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv3d
        self.conv_fn = conv_fn

    def conv_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self.reversed_padding_x2, mode=self.padding_mode)
            return self.conv_fn(x, weight, bias, self.stride,
                                (0, 0, 0), self.dilation, self.groups)
        return self.conv_fn(x, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)


class _ConvTransposeNd(_ConvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 output_padding,
                 groups,
                 bias: Optional[bool],
                 padding_mode: str) -> None:
        if padding_mode != "zeros":
            raise ValueError("[ERROR:NN] Only 'zeros' padding is supported for ConvTranspose.")

        super(_ConvTransposeNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                               True, output_padding, groups, bias, padding_mode)

    def conv_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                  output_size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        output_padding = self._output_padding(x, output_size, self.stride, self.padding,
                                              self.kernel_size, self.dilation)
        return self.conv_fn(x, weight, bias, self.stride, self.padding,
                            output_padding, self.groups, self.dilation)

    def forward(self, x: torch.Tensor, output_size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        bias = self.bias() if (self.bias is not None) else None
        return self.conv_core(x, self.weight(), bias, output_size)

    def _output_padding(self, x: torch.Tensor, output_size: Optional[Tuple[int, ...]],
                        stride: Tuple[int, ...], padding: Tuple[int, ...],
                        kernel_size: Tuple[int, ...], dilation: Tuple[int, ...]) -> Tuple[int, ...]:
        if output_size is None:
            ret = to_single_tuple(self.output_padding)  # converting to list if was not already
        else:
            k = x.ndim - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    f"[ERROR:NN] ConvTranspose output_size must have {k} or {k + 2} elements (got {len(output_size)})")

            min_sizes = []
            max_sizes = []
            for d in range(k):
                dim_size = ((x.shape[d + 2] - 1) * stride[d] - 2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(f"[ERROR:NN] ConvTranspose requested an output size of {output_size}, "
                                     f"but valid sizes range from {min_sizes} to {max_sizes} "
                                     f"(for an input of {x.shape[2:]})")

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class ConvTranspose1d(_ConvTransposeNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias: Optional[bool] = True,
                 dilation=1,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_single_tuple(kernel_size)
        stride = to_single_tuple(stride)
        padding = to_single_tuple(padding)
        dilation = to_single_tuple(dilation)

        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              output_padding, groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv_transpose1d
        self.conv_fn = conv_fn


class ConvTranspose2d(_ConvTransposeNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias: Optional[bool] = True,
                 dilation=1,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_pair_tuple(kernel_size)
        stride = to_pair_tuple(stride)
        padding = to_pair_tuple(padding)
        dilation = to_pair_tuple(dilation)

        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              output_padding, groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv_transpose2d
        self.conv_fn = conv_fn


class ConvTranspose3d(_ConvTransposeNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias: Optional[bool] = True,
                 dilation=1,
                 padding_mode: str = "zeros",
                 *, conv_fn=None) -> None:
        kernel_size = to_triple_tuple(kernel_size)
        stride = to_triple_tuple(stride)
        padding = to_triple_tuple(padding)
        dilation = to_triple_tuple(dilation)

        super(ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              output_padding, groups, bias, padding_mode)
        if conv_fn is None:
            conv_fn = F.conv_transpose3d
        self.conv_fn = conv_fn
