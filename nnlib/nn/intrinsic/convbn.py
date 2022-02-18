from typing import Tuple
import torch
import math
import torch.nn as tnn

from nnlib.nn.modules import BaseModule, Conv2d, BatchNorm2d, ReLU, ReLU6
from nnlib.nn.parameter import ParameterModule


class ConvBn2d(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size, stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 padding_mode: str = "zeros",
                 merged: bool = False, *,
                 conv_fn=None, norm_fn=None):
        super(ConvBn2d, self).__init__()
        self.is_merged = merged

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                           bias=False, padding_mode=padding_mode, conv_fn=conv_fn)
        self.bn = BatchNorm2d(out_channels, eps, momentum, affine=affine,
                              track_running_stats=track_running_stats)

        self._initialize_parameters()
        self._merged_permanently = False

    def _initialize_parameters(self):
        tnn.init.kaiming_normal_(self.conv.weight.data, a=math.sqrt(5), mode="fan_out", nonlinearity="leaky_relu")

    def freeze(self, flag: bool = True):
        if self.bn is not None:
            _ = self.bn.freeze(flag)
        return self

    def merge_permanently(self):
        merged_w, merged_b = self._merge_conv_bn_eval()
        del self.conv.weight  # ParameterModule
        self.conv.weight = ParameterModule(merged_w)
        self.conv.bias = ParameterModule(merged_b)
        del self.bn
        self.bn = None
        self._merged_permanently = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_merged:  # default
            x = self.conv(x)
            x = self.bn(x)
            return x
        elif not self._merged_permanently:  # merge flag is ON, but not merged permanently
            if self.training and self.bn.training:  # training & bn not frozen
                y = self.conv(x)
                y_mean = torch.mean(y, dim=[0, 2, 3])
                y_var = torch.var(y, dim=[0, 2, 3], unbiased=False)
                y_size = y.shape[0] * y.shape[2] * y.shape[3]

                merged_w, merged_b = self._merge_conv_bn_train(y_mean, y_var)  # batch stats
                self._update_bn_stat(y_mean, y_var, y_size)
            elif self.training:  # training and bn frozen
                y = self.conv(x)
                y_mean = torch.mean(y, dim=[0, 2, 3])
                y_var = torch.var(y, dim=[0, 2, 3], unbiased=False)
                # y_size = y.shape[0] * y.shape[2] * y.shape[3]
                scale, offset = self._compute_correction_coefficients(y_mean, y_var)
                scale = scale.detach()
                offset = offset.detach()

                # keep BN functionality
                merged_w, merged_b = self._merge_conv_bn_train(y_mean, y_var)  # batch stats
                merged_w = merged_w / scale  # running stats
                merged_b = merged_b - offset  # running stats
            else:
                merged_w, merged_b = self._merge_conv_bn_eval()
            return self.conv.conv_core(x, merged_w, merged_b)  # merged conv run
        else:  # merged permanently
            return self.conv(x)

    def get_merged_params(self):
        return self._merge_conv_bn_eval()

    def _compute_correction_coefficients(self, mean: torch.Tensor, var: torch.Tensor):
        # correct weight and bias to match batch statistics.
        # weight(batch)" = weight(running) * scale
        # bias(batch)" = bias(running) + offset
        # correction coefficients does not flow gradient...?
        running_mean = self.bn.running_mean().clone().detach() if (self.bn.running_mean is not None) else 0.0
        running_var = self.bn.running_var().clone().detach() if (self.bn.running_var is not None) else 1.0
        running_inv_std = torch.rsqrt(running_var + self.bn.eps) if (self.bn.running_var is not None) else 1.0

        inv_std = torch.rsqrt(var + self.bn.eps)
        scale = (inv_std / running_inv_std).view(-1, 1, 1, 1)

        gamma = self.bn.weight() if (self.bn.weight is not None) else 1.0
        offset = gamma * (running_mean * running_inv_std - mean * inv_std)
        return scale, offset

    def _merge_conv_bn_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns merged Conv weight and bias using running mean/var"""
        w = self.conv.weight()
        assert self.conv.bias is None
        gamma = self.bn.weight() if (self.bn.weight is not None) else 1.0
        beta = self.bn.bias() if (self.bn.bias is not None) else 0.0

        # TODO actually we need to compute input mean/var if track_running_stats is False.
        running_mean = self.bn.running_mean().clone().detach() if (self.bn.running_mean is not None) else 0.0
        running_var = self.bn.running_var().clone().detach() if (self.bn.running_var is not None) else 1.0
        running_inv_std = torch.rsqrt(running_var + self.bn.eps) if (self.bn.running_var is not None) else 1.0

        # y = gamma * ((x * w - mean) / std) + beta
        # y = x * (gamma / std) * w + beta - (gamma / std) * mean
        bn_weight = gamma * running_inv_std
        merged_w = w * bn_weight.view(-1, 1, 1, 1)

        merged_b = beta - bn_weight * running_mean
        return merged_w, merged_b

    def _merge_conv_bn_train(self, mean: torch.Tensor, var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns merged Conv weight and bias using input mean/var"""
        w = self.conv.weight()
        assert self.conv.bias is None
        gamma = self.bn.weight() if (self.bn.weight is not None) else 1.0
        beta = self.bn.bias() if (self.bn.bias is not None) else 0.0
        inv_std = torch.rsqrt(var + self.bn.eps)

        # y = gamma * ((x * w - mean) / std) + beta
        # y = x * (gamma / std) * w + beta - (gamma / std) * mean
        bn_weight = gamma * inv_std
        merged_w = w * bn_weight.view(-1, 1, 1, 1)
        merged_b = beta - bn_weight * mean

        return merged_w, merged_b

    def _update_bn_stat(self, mean: torch.Tensor, var: torch.Tensor, size: int) -> None:
        # update running statistics
        # m" = m * (1 - momentum) + new_m * momentum
        # m" = m + momentum * (new_m - m)
        with torch.no_grad():
            if self.bn.running_mean is not None:
                self.bn.running_mean.data.add_(mean.detach() - self.bn.running_mean.data,
                                               alpha=self.bn.momentum)
            if self.bn.running_var is not None:
                unbiased_var = var * float(size / (size - 1))
                self.bn.running_var.data.add_(unbiased_var.detach() - self.bn.running_var.data,
                                              alpha=self.bn.momentum)


class ConvBnReLU2d(ConvBn2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size, stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps: float = 1e-5
                 , momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 padding_mode: str = "zeros",
                 merged: bool = False, *,
                 conv_fn=None, norm_fn=None, act_layer=None):
        super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, eps, momentum, affine, track_running_stats, padding_mode,
                                           merged, conv_fn=conv_fn, norm_fn=norm_fn)
        if act_layer is None:
            act_layer = ReLU
        self.act_layer = act_layer()
        if hasattr(self.act_layer, "inplace"):
            self.act_layer.inplace = True
        self._initialize_parameters_act()

    def _initialize_parameters_act(self):
        if isinstance(self.act_layer, (ReLU, ReLU6)):
            tnn.init.kaiming_normal_(self.conv.weight.data, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(ConvBnReLU2d, self).forward(x)
        x = self.act_layer(x)
        return x


class ConvReLU2d(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = "zeros", *,
                 conv_fn=None, act_layer=None):
        super(ConvReLU2d, self).__init__()

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                           bias=bias, padding_mode=padding_mode, conv_fn=conv_fn)

        if act_layer is None:
            act_layer = ReLU
        self.act_layer = act_layer()
        self._initialize_parameters_act()

    def _initialize_parameters_act(self):
        if isinstance(self.act_layer, (ReLU, ReLU6)):
            tnn.init.kaiming_normal_(self.conv.weight.data, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act_layer(x)
        return x


class BnConv2d(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 padding_mode: str = "zeros",
                 merged: bool = False, *,
                 conv_fn=None, norm_fn=None):
        super(BnConv2d, self).__init__()
        self.is_merged = merged

        self.bn = BatchNorm2d(in_channels, eps, momentum, affine=affine,
                              track_running_stats=track_running_stats)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                           bias=True, padding_mode=padding_mode, conv_fn=conv_fn)

        self._initialize_parameters()

    def _initialize_parameters(self):
        tnn.init.kaiming_normal_(self.conv.weight.data, a=math.sqrt(5), mode="fan_out", nonlinearity="leaky_relu")

    def freeze(self, flag: bool = True):
        if self.bn is not None:
            _ = self.bn.freeze(flag)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_merged:
            x = self.bn(x)
            x = self.conv(x)
            return x
        else:
            raise NotImplementedError


class BnReLUConv2d(BnConv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 padding_mode: str = "zeros",
                 merged: bool = False, *,
                 conv_fn=None, norm_fn=None, act_layer=None):
        super(BnReLUConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, eps, momentum, affine, track_running_stats, padding_mode,
                                           merged, conv_fn=conv_fn, norm_fn=norm_fn)
        if act_layer is None:
            act_layer = ReLU
        self.act_layer = act_layer()
        if hasattr(self.act_layer, "inplace"):
            self.act_layer.inplace = True
        self._initialize_parameters_act()

    def _initialize_parameters_act(self):
        if isinstance(self.act_layer, (ReLU, ReLU6)):
            tnn.init.kaiming_normal_(self.conv.weight.data, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_merged:
            x = self.bn(x)
            x = self.act_layer(x)
            x = self.conv(x)
            return x
        else:
            raise NotImplementedError
