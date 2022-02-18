import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.parameter import ParameterModule, ParameterModuleWithOffset


class LayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 *, norm_fn=None):
        super(LayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = ParameterModuleWithOffset(torch.zeros(normalized_shape, dtype=torch.float32), offset=1.0)
            self.bias = ParameterModule(torch.zeros(normalized_shape, dtype=torch.float32))
        else:
            self.weight = self.bias = None
        if norm_fn is None:
            norm_fn = F.layer_norm
        self.norm_fn = norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_fn(
            x, list(self.normalized_shape),
            self.weight() if (self.weight is not None) else None,
            self.bias() if (self.bias is not None) else None,
            self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class GroupNorm(BaseModule):

    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 *, norm_fn=None):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = ParameterModuleWithOffset(torch.zeros((num_channels,), dtype=torch.float32), offset=1.0)
            self.bias = ParameterModule(torch.zeros((num_channels,), dtype=torch.float32))
        else:
            self.weight = self.bias = None
        if norm_fn is None:
            norm_fn = F.group_norm
        self.norm_fn = norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_fn(
            x, self.num_groups,
            self.weight() if (self.weight is not None) else None,
            self.bias() if (self.bias is not None) else None,
            self.eps)

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


class GroupLayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 groups: int = 1,
                 *, norm_fn=None):
        super(GroupLayerNorm, self).__init__()
        if not isinstance(normalized_shape, int):
            raise ValueError(f"[ERROR:GroupLN] Currently only support last-dim LN for GroupLN.")

        if normalized_shape % groups != 0:
            raise ValueError(f"[ERROR:GroupLN] Normalized shape {normalized_shape} "
                             f"is not divisible by groups {groups}.")
        self.groups = groups

        self.normalized_shape = (normalized_shape // groups,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = ParameterModuleWithOffset(torch.zeros(normalized_shape, dtype=torch.float32), offset=1.0)
            self.bias = ParameterModule(torch.zeros(normalized_shape, dtype=torch.float32))
        else:
            self.weight = self.bias = None

        if norm_fn is None:
            norm_fn = F.layer_norm
        self.norm_fn = norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = tuple(x.shape)
        g = self.groups
        d_per_g = self.normalized_shape[-1]
        assert x_shape[-1] == g * d_per_g

        x_view = x.view(*x_shape[:-1], g, d_per_g)

        y = self.norm_fn(x_view, [d_per_g], None, None, eps=self.eps)
        y = y.view(x_shape)
        if self.elementwise_affine:
            y = y * self.weight() + self.bias()
        return y

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}," \
               f" elementwise_affine={self.elementwise_affine}, groups={self.groups}"


class ScaleOnlyLayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps: float = 1e-5,
                 *, norm_fn=None):
        super(ScaleOnlyLayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = ParameterModuleWithOffset(torch.zeros(normalized_shape, dtype=torch.float32), offset=1.0)
        if norm_fn is None:
            norm_fn = F.layer_norm
        self.norm_fn = norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_fn(
            x, list(self.normalized_shape),
            self.weight() if (self.weight is not None) else None,
            None,
            self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    @staticmethod
    def from_base(module: LayerNorm):
        cls = ScaleOnlyLayerNorm(
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            norm_fn=module.norm_fn
        )
        cls.weight.data = torch.nn.Parameter(module.weight.data.clone().detach())
        return cls
