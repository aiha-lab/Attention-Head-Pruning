import torch
import torch.nn.functional as F

from nnlib.nn.modules.batchnorm import _BatchNorm


class _InstanceNorm(_BatchNorm):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = False,
                 track_running_stats: bool = False,
                 *, norm_fn=None) -> None:
        if norm_fn is None:
            norm_fn = F.instance_norm
        super(_InstanceNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats,
                                            norm_fn=norm_fn)

    def _check_input_dim(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        return self.norm_fn(
            x,
            self.running_mean() if (self.running_mean is not None) else None,
            self.running_var() if (self.running_var is not None) else None,
            self.weight() if (self.weight is not None) else None,
            self.bias() if (self.bias is not None) else None,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class InstanceNorm1d(_InstanceNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 2 and x.ndim != 3:
            raise ValueError(f"[ERROR:NN] InstanceNorm1d expect 2D/3D input, but got {x.shape}.")


class InstanceNorm2d(_InstanceNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"[ERROR:NN] InstanceNorm2d expect 4D input, but got {x.shape}.")


class InstanceNorm3d(_InstanceNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"[ERROR:NN] InstanceNorm3d expect 5D input, but got {x.shape}.")
