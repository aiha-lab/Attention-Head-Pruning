from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.parameter import ParameterModule


class Linear(BaseModule):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 *, weight_transposed: bool = False,
                 linear_fn=None) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        if linear_fn is None:
            linear_fn = F.linear
        self.linear_fn = linear_fn

        if not weight_transposed:
            self.weight = ParameterModule(torch.empty(out_features, in_features))
        else:
            self.weight = ParameterModule(torch.empty(in_features, out_features))
        self.weight_transposed = weight_transposed

        if bias:
            self.bias = ParameterModule(torch.zeros(out_features, ))
        else:
            self.bias = None

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
        if self.bias is not None:
            # bound = 1.0 / math.sqrt(self.in_features)
            # nn.init.uniform_(self.bias.data, -bound, bound)
            nn.init.zeros_(self.bias.data)

    def linear_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.linear_fn(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias() if (self.bias is not None) else None
        if not self.weight_transposed:
            return self.linear_core(x, self.weight(), bias)
        else:
            return self.linear_core(x, self.weight().t(), bias)

    def extra_repr(self) -> str:
        s = f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"
        if self.weight_transposed:
            s += f", weight_transposed=True"
        return s


# Currently does not seem to use this class.
# class BiasAdd(BaseModule):
#
#     def __init__(self, num_features: int):
#         super(BiasAdd, self).__init__()
#         self.num_features = num_features
#
#         self.bias = ParameterModule(torch.zeros(num_features, ))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.bias  # broadcast to latest dim


class GroupLinear(BaseModule):
    """Used in DeLighT
    https://github.com/sacmehta/delight/blob/master/fairseq/delight_modules/nn_functions.py
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 groups: int = 1,
                 *, linear_fn=None) -> None:
        super(GroupLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        if out_features % groups != 0:
            raise ValueError(f"[ERROR:LINEAR] Output dimension {out_features} is not divisible by groups {groups}")
        if in_features % groups != 0:
            raise ValueError(f"[ERROR:LINEAR] Input dimension {in_features} is not divisible by groups {groups}")
        self.groups = groups

        if linear_fn is None:
            linear_fn = F.linear
        self.linear_fn = linear_fn

        self.weight = ParameterModule(torch.empty(groups, in_features // groups, out_features // groups))
        if bias:
            self.bias = ParameterModule(torch.zeros(groups, 1, out_features // groups))
        else:
            self.bias = None

        self._initialize_parameters()

    def _initialize_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features // self.groups)
        nn.init.uniform_(self.weight.data, -bound, bound)
        if self.bias is not None:
            # bound = 1.0 / math.sqrt(self.in_features)
            # nn.init.uniform_(self.bias.data, -bound, bound)
            nn.init.zeros_(self.bias.data)

    def linear_core(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x:      (..., input_dim) -> (..., groups, input_dim // groups)
        w:      (groups, input_dim // groups, output_dim // groups)
        b:      (groups, output_dim // groups)
        y:      (..., output_dim)
        """
        x_shape = tuple(x.shape)
        assert x_shape[-1] == self.in_features
        y_shape = x_shape[:-1] + (self.out_features,)

        g = self.groups
        in_d_per_g = self.in_features // g

        x = x.view(-1, g, in_d_per_g).transpose(0, 1)  # (g, -1, in_d // g)
        y = torch.bmm(x, weight)  # (g, -1, out_d // g)
        if bias is not None:
            y = y + bias

        y = y.transpose(1, 0).reshape(*y_shape)  # (-1, g, out_d // g) -> (..., out_d)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias() if (self.bias is not None) else None
        return self.linear_core(x, self.weight(), bias)

    def extra_repr(self) -> str:
        s = f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, groups={self.groups}"
        return s
