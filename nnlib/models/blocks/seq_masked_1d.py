from typing import Optional
import torch
import torch.nn.functional as F

from nnlib.nn import Conv1d, Conv2d, BatchNorm1d, BatchNorm2d
from nnlib.nn.functional import masked_batch_norm_func, masked_sync_batch_norm_func


class SeqMaskedConv1d(Conv1d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias: Optional[bool] = True,
                 *, partial: bool = False,
                 conv_fn=None) -> None:

        # we force zero padding, and output seq_length is same as input seq_length by proper padding.
        padding_mode = "zeros"  # force set
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        super(SeqMaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias=bias, padding_mode=padding_mode, conv_fn=conv_fn)

        self.partial = partial
        if partial:
            dummy_weight = torch.ones(1, 1, kernel_size, dtype=torch.float32) / kernel_size
        else:
            dummy_weight = None
        self.register_buffer("dummy_weight", dummy_weight, persistent=False)

    @torch.no_grad()
    def _generate_scale_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask:       (batch_size, 1, seq_length)
        """
        device = self.dummy_weight.device
        mask = mask.float().to(device)
        scale = F.conv1d(mask, self.dummy_weight, bias=None,
                         stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        scale = mask / scale.add_(1e-6)
        scale = scale.clamp_(1.0, 2.0)
        return scale  # (batch_size, 1, seq_length)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:          (batch_size, dim, seq_length)
        mask:       (batch_size, seq_length)
        """
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("[ERROR:MODEL] MaskedConv1d mask should be 2D.")
            mask = mask.unsqueeze(1)  # (b, 1, s)
            if mask.shape != (x.shape[0], 1, x.shape[2]):
                raise ValueError(f"[ERROR:MODEL] MaskedConv1d mask shape {mask.shape} mismatch to input {x.shape}")
            x = x * mask

        y = super(SeqMaskedConv1d, self).forward(x)
        if (mask is not None) and self.partial:
            scale = self._generate_scale_mask(mask).detach_()
            y = y * scale
        return y


class SeqMaskedConv2d(Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias: Optional[bool] = True,
                 *, partial: bool = False,
                 conv_fn=None) -> None:

        # we force zero padding, and output seq_length is same as input seq_length by proper padding.
        padding_mode = "zeros"  # force set
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        super(SeqMaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias=bias, padding_mode=padding_mode, conv_fn=conv_fn)

        self.partial = partial
        if partial:
            dummy_weight = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32) / (kernel_size * kernel_size)
        else:
            dummy_weight = None
        self.register_buffer("dummy_weight", dummy_weight, persistent=False)

    @torch.no_grad()
    def _generate_scale_mask(self, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """
        mask:       (batch_size, 1, seq_length, 1)
        """
        b, _, s, _ = mask.shape
        assert mask.shape == (b, 1, s, 1)

        device = self.dummy_weight.device
        mask = mask.float().to(device).expand(b, 1, s, dim)
        scale = F.conv2d(mask, self.dummy_weight, bias=None,
                         stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        scale = mask / scale.add_(1e-6)
        scale = scale.clamp_(1.0, 2.0)
        return scale  # (batch_size, 1, seq_length, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:          (batch_size, dim, seq_length)
        mask:       (batch_size, seq_length)
        """
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("[ERROR:MODEL] MaskedConv2d mask should be 2D.")
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (b, 1, s, 1)
            if mask.shape != (x.shape[0], 1, x.shape[2], 1):
                raise ValueError(f"[ERROR:MODEL] MaskedConv2d mask shape {mask.shape} mismatch to input {x.shape}")
            x = x * mask

        y = super(SeqMaskedConv2d, self).forward(x)
        if (mask is not None) and self.partial:
            scale = self._generate_scale_mask(mask, y.shape[-1]).detach_()
            y = y * scale
        return y


class SeqMaskedBatchNorm1d(BatchNorm1d):
    """https://arxiv.org/pdf/1510.01378.pdf"""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_input_dim(x)
        x = x.float()
        b, c, s = x.shape
        # ------------------------------------------------------------------------------------------------ #
        # check mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("[ERROR:MODEL] MaskedBN1d mask should be 2D.")
            mask = mask.unsqueeze(1)  # (b, 1, s)
            if mask.shape != (b, 1, s):
                raise ValueError(f"[ERROR:MODEL] MaskedBN1d mask shape {mask.shape} mismatch to input {x.shape}")

        if (mask is None) or (not self.training):
            if mask is not None:
                x = x * mask
            return super(SeqMaskedBatchNorm1d, self).forward(x)

        running_mean = self.running_mean() if (self.running_mean is not None) else None
        running_var = self.running_var().clamp_min_(1e-6) if (self.running_var is not None) else None

        if not self.sync_bn:
            return masked_batch_norm_func(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                mask,
                running_mean,
                running_var,
                self.momentum,
                self.eps
            )
        else:
            return masked_sync_batch_norm_func(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                mask,
                running_mean,
                running_var,
                self.momentum,
                self.eps
            )


class SeqMaskedBatchNorm2d(BatchNorm2d):
    """https://arxiv.org/pdf/1510.01378.pdf"""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_input_dim(x)
        x = x.float()
        b, c, s, d = x.shape
        # ------------------------------------------------------------------------------------------------ #
        # check mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("[ERROR:MODEL] MaskedBN2d mask should be 2D.")
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(b, 1, s, d)  # (b, 1, s, 1)
            if mask.shape != (b, 1, s, d):
                raise ValueError(f"[ERROR:MODEL] MaskedBN2d mask shape {mask.shape} mismatch to input {x.shape}")

        if (mask is None) or (not self.training):
            if mask is not None:
                x = x * mask
            return super(SeqMaskedBatchNorm2d, self).forward(x)

        running_mean = self.running_mean() if (self.running_mean is not None) else None
        running_var = self.running_var().clamp_min_(1e-6) if (self.running_var is not None) else None

        if not self.sync_bn:
            return masked_batch_norm_func(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                mask,
                running_mean,
                running_var,
                self.momentum,
                self.eps
            )
        else:
            return masked_sync_batch_norm_func(
                x,
                self.weight() if (self.weight is not None) else None,
                self.bias() if (self.bias is not None) else None,
                mask,
                running_mean,
                running_var,
                self.momentum,
                self.eps
            )
