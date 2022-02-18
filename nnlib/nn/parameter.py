from typing import Optional
import torch
import torch.nn as tnn

from nnlib.nn.modules.module import BaseModule


class _DataModule(BaseModule):
    """Module which intend to contain only one tensor."""

    def __init__(self, data: Optional[torch.Tensor], is_param: bool = True, **kwargs):
        super(_DataModule, self).__init__()

        if is_param:
            if data is not None:
                self.data = tnn.Parameter(data, requires_grad=kwargs.get("requires_grad", True))
            else:  # None
                self.register_parameter("data", None)
        else:  # buffer
            self.register_buffer("data", data, persistent=kwargs.get("persistent", True))

    def forward(self) -> Optional[torch.Tensor]:
        return self.data

    def __call__(self):
        # hook is not supported.
        return self.forward()

    def register_forward_hook(self, hook):
        raise RuntimeError("[ERROR:PARAM] hook is not supported for ParameterModule/BufferModule.")

    def register_forward_pre_hook(self, hook):
        raise RuntimeError("[ERROR:PARAM] hook is not supported for ParameterModule/BufferModule.")

    def register_backward_hook(self, hook):
        raise RuntimeError("[ERROR:PARAM] hook is not supported for ParameterModule/BufferModule.")

    def register_full_backward_hook(self, hook):
        raise RuntimeError("[ERROR:PARAM] hook is not supported for ParameterModule/BufferModule.")

    @property
    def ndim(self) -> int:
        return self.data.ndim if (self.data is not None) else 0

    @property
    def shape(self):
        return self.data.shape if (self.data is not None) else None

    def size(self):
        return self.data.size() if (self.data is not None) else None

    def numel(self) -> Optional[int]:
        return self.data.numel() if (self.data is not None) else None

    @property
    def grad(self):
        return self.data.grad if (self.data is not None) else None

    @property
    def device(self):
        return self.data.device if (self.data is not None) else None

    @property
    def dtype(self):
        return self.data.dtype if (self.data is not None) else None

    @property
    def requires_grad(self):
        return self.data.requires_grad if (self.data is not None) else None

    def requires_grad_(self, requires_grad: bool = True):
        if self.data is None:
            raise ValueError(
                f"[ERROR:PARAM] Trying to set requires_grad={requires_grad} on {self.name}, but target is None.")
        return self.data.requires_grad_(requires_grad)

    def extra_repr(self) -> str:
        if self.data is None:
            return f"None"
        s = f"shape={tuple(self.shape)}"
        if self.dtype != torch.float32:
            s += f", dtype={self.dtype}"
        if self.device != torch.device("cpu"):
            s += f", device={self.device}"
        return s


class ParameterModule(_DataModule):

    def __init__(self, data: Optional[torch.Tensor], requires_grad: bool = True):
        super(ParameterModule, self).__init__(data, is_param=True, requires_grad=requires_grad)


class BufferModule(_DataModule):

    def __init__(self, data: Optional[torch.Tensor], persistent: bool = True, ):
        super(BufferModule, self).__init__(data, is_param=False, persistent=persistent)


class ParameterModuleWithOffset(ParameterModule):

    def __init__(self,
                 data: torch.Tensor,
                 offset: float,
                 requires_grad: bool = True):
        if data is None:
            raise ValueError("[ERROR:PARAM] ParameterModuleWithOffset data must be torch.Tensor, got None.")
        super(ParameterModuleWithOffset, self).__init__(data, requires_grad=requires_grad)
        self.offset = offset

    def forward(self) -> torch.Tensor:
        return self.data + self.offset

    @staticmethod
    def from_base(module: ParameterModule, *, offset=0.0):
        return ParameterModuleWithOffset(
            data=module.data.data,
            offset=offset,
            requires_grad=module.requires_grad,
        )


class ParameterModuleWithWeightStandardization(ParameterModule):

    def __init__(self,
                 data: torch.Tensor,
                 eps: float = 1e-5,
                 axis: int = 0,
                 requires_grad: bool = True):
        super(ParameterModuleWithWeightStandardization, self).__init__(data, requires_grad=requires_grad)
        self.axis = axis
        self.eps = eps

    def forward(self) -> Optional[torch.Tensor]:
        if self.ndim <= 1:  # do not apply standardization
            return self.data

        if self.axis < 0:
            self.axis = self.axis + self.ndim
        if not (0 <= self.axis < self.ndim):
            raise ValueError(f"[ERROR:PARAM] Invalid axis {self.axis} for weight standardization, shape {self.shape}.")

        dim_list = list(range(0, self.ndim))
        dim_list[self.axis] = 0
        dim_list[0] = self.axis

        mean = torch.mean(self.data, dim=dim_list[1:], keepdim=True)
        var = torch.var(self.data, dim=dim_list[1:], keepdim=True, unbiased=False).add(self.eps)
        inv_std = torch.rsqrt(var)
        return (self.data - mean) * inv_std

    @staticmethod
    def from_base(module: ParameterModule, *, eps: float = 1e-5, axis: int = 0):
        return ParameterModuleWithWeightStandardization(
            data=module.data.clone().detach(),
            eps=eps,
            axis=axis,
            requires_grad=module.requires_grad,
        )


class ParameterModuleWithNoise(ParameterModule):

    def __init__(self,
                 data: torch.Tensor,
                 noise_mean: float = 0.0,
                 noise_std: float = 0.01,
                 requires_grad: bool = True):
        if data is None:
            raise ValueError("[ERROR:PARAM] ParameterModuleWithNoise data must be torch.Tensor, got None.")
        super(ParameterModuleWithNoise, self).__init__(data, requires_grad=requires_grad)
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    @torch.no_grad()
    def _generate_noise(self):
        noise = torch.randn_like(self.data)
        return self.noise_mean + noise * self.noise_std

    def forward(self) -> torch.Tensor:
        if not self.training:
            return self.data

        if (self.noise_mean == 0.0) and (self.noise_std == 0.0):
            return self.data

        noise = self._generate_noise()
        return self.data + noise

    @staticmethod
    def from_base(module: ParameterModule, *, mean: float = 0.0, std: float = 0.01):
        return ParameterModuleWithNoise(
            data=module.data.data,
            noise_mean=mean,
            noise_std=std,
            requires_grad=module.requires_grad,
        )
