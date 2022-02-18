from typing import Optional
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.parameter import ParameterModule, BufferModule, ParameterModuleWithOffset
from nnlib.nn.functional import sync_batch_norm_func

from nnlib.utils.dist_utils import get_world_size


class _NormBase(BaseModule):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 norm_fn=None) -> None:
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps

        if momentum is None:
            raise ValueError(f"[ERROR:NN] BatchNorm does not support momentum=None (unlike original impl.)")
        if not (0 <= momentum <= 1):
            raise ValueError(f"[ERROR:NN] BatchNorm momentum should be in [0, 1) but got {momentum}.")
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self._frozen = False

        if self.affine:
            # self.weight = ParameterModule(torch.ones(num_features, dtype=torch.float32))
            self.weight = ParameterModuleWithOffset(torch.zeros(num_features, dtype=torch.float32), offset=1.0)
            self.bias = ParameterModule(torch.zeros(num_features, dtype=torch.float32))
        else:
            self.weight = self.bias = None

        if self.track_running_stats:
            self.running_mean = BufferModule(torch.zeros(num_features, dtype=torch.float32))
            self.running_var = BufferModule(torch.ones(num_features, dtype=torch.float32))
            self.num_batches_tracked = BufferModule(torch.as_tensor(0, dtype=torch.long))
        else:
            self.running_mean = self.running_var = self.num_batches_tracked = None

    def freeze(self, flag: bool = True):
        # Do not update statistics, same as .eval()
        self._frozen = flag
        self.training = (not flag)
        return self

    def train(self, mode: bool = True):
        if self._frozen:
            self.training = False
        else:
            self.training = mode

        for module in self.children():
            module.train(mode)
        return self

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.data.zero_()
            self.running_var.data.fill_(1.0)
            self.num_batches_tracked.data.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(0.0)  # offset 1.0 exist
            self.bias.data.fill_(0)

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
        if not self.affine:
            s += f", affine=False"
        if not self.track_running_stats:
            s += f", track_running_stats=False"
        return s


class _BatchNorm(_NormBase):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 sync_bn: bool = False) -> None:
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.sync_bn = sync_bn and (get_world_size() > 1)
        if self.sync_bn:
            self.print(f"[LOG:BN] SyncBN enabled. (num_features: {num_features}, momentum: {momentum}, eps: {eps})")

    def _check_input_dim(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        x = x.float()

        # ------------------------------------------------------------------------------------------------ #
        # training flag
        # decide whether to use mini-batch statistics: during training OR during eval but running stats are None.
        if self.training:
            bn_training = True
        else:  # eval, for most case: False
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # ------------------------------------------------------------------------------------------------ #
        # if not track_running_stats, use batch stat for both train/eval.
        # buffers are only updated when they are to be tracked and in training mode.
        # so the buffers should be passed only when update occur (training & tracked), or used for normalization (eval)

        running_mean = self.running_mean() if (self.running_mean is not None) else None
        running_var = self.running_var().clamp_min_(1e-6) if (self.running_mean is not None) else None

        if (not self.sync_bn) or (not bn_training):
            return torch.batch_norm(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                running_mean if (not self.training or self.track_running_stats) else None,
                running_var if (not self.training or self.track_running_stats) else None,
                bn_training,
                self.momentum,
                self.eps,
                cudnn.enabled
            )
        else:  # sync_bn and self.training
            return sync_batch_norm_func(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                running_mean if (not self.training or self.track_running_stats) else None,
                running_var if (not self.training or self.track_running_stats) else None,
                self.momentum,
                self.eps
            )

        # return self.norm_fn(
        #     x,
        #     # If buffers are not to be tracked, ensure that they won't be updated
        #     self.running_mean() if (not self.training or self.track_running_stats) else None,
        #     self.running_var() if (not self.training or self.track_running_stats) else None,
        #     self.weight() if (self.weight is not None) else None,
        #     self.bias() if (self.bias is not None) else None,
        #     bn_training,
        #     self.momentum,
        #     self.eps)


class BatchNorm1d(_BatchNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 2 and x.ndim != 3:
            raise ValueError(f"[ERROR:NN] BatchNorm1d expect 2D/3D input, but got {x.shape}.")


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"[ERROR:NN] BatchNorm2d expect 4D input, but got {x.shape}.")


class BatchNorm3d(_BatchNorm):

    def _check_input_dim(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"[ERROR:NN] BatchNorm3d expect 5D input, but got {x.shape}.")


class GhostBatchNormWrapper(BaseModule):

    def __init__(self, bn_module: _BatchNorm, ghost_batch_size: int = 0):
        super(GhostBatchNormWrapper, self).__init__()
        self.module = bn_module
        if bn_module.name is not None:
            self.module._name = bn_module.name + ".module"
        self.ghost_batch_size = ghost_batch_size

    def freeze(self, flag: bool = True):
        self.module.freeze(flag=flag)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.ghost_batch_size <= 1):  # not applicable
            return self.module.forward(x)

        x_shape = x.shape
        batch_size = x_shape[0]
        if batch_size % self.ghost_batch_size != 0:
            raise ValueError(f"[ERROR:NN] GhostBN should have "
                             f"batch_size {batch_size} % ghost size {self.ghost_batch_size} == 0.")
        res = []
        num_ghost = batch_size // self.ghost_batch_size
        x_split = x.view(num_ghost, self.ghost_batch_size, *x_shape[1:])
        for i in range(self.ghost_batch_size):
            res.append(self.module.forward(x_split[i]))
        res = torch.cat(res, dim=0).view(x_shape)
        return res
