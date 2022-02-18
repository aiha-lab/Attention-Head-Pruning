from typing import Optional
import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule


class _DropoutNd(BaseModule):

    def __init__(self,
                 drop_prob: float = 0.5,
                 inplace: bool = False):
        super(_DropoutNd, self).__init__()
        if not (0 <= drop_prob < 1):
            raise ValueError(f"[ERROR:NN] Dropout probability should be in [0, 1), but got p={drop_prob}.")
        self.p = drop_prob
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class Dropout(_DropoutNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout2d(x, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout3d(x, self.p, self.training, self.inplace)


class AlphaDropout(_DropoutNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.alpha_dropout(x, self.p, self.training)


class FeatureAlphaDropout(_DropoutNd):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.feature_alpha_dropout(x, self.p, self.training)


class SequenceDropout(_DropoutNd):
    """Behaves as normal dropout if axial_drop=False, else use fixed mask through axis."""

    def __init__(self,
                 drop_prob: float = 0.5,
                 axial_drop: bool = False,
                 inplace: bool = False,
                 *, axis: int = 1):
        super(SequenceDropout, self).__init__(drop_prob=drop_prob, inplace=inplace)

        # default axis is 1 because we recommend sequence-dimension dropout, which is often 1
        # mask will be shared through selected axis. (ex) all time sequence will see same mask.
        self.axial_drop = axial_drop
        self.axis = axis

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (not self.training) or (self.p <= 0):
            return x

        if not self.axial_drop:
            return F.dropout(x, self.p, self.training, self.inplace)

        if mask is None:
            mask = self.generate_mask(x)
        return x * mask

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = list(x.shape)  # (B, S, D)
        x_shape[self.axis] = 1  # (B, 1, D)

        keep_p = 1.0 - self.p
        mask = torch.bernoulli(torch.ones(x_shape, device=x.device, dtype=x.dtype, requires_grad=False), p=keep_p)
        mask = mask.div_(keep_p)
        return mask


class BatchDropout(_DropoutNd):

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (not self.training) or (self.p <= 0):
            return x

        if mask is None:
            mask = self.generate_mask(x)
        return x * mask

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        shape = (batch_size,) + (1,) * (x.ndim - 1)  # (b, 1, 1, ...)

        keep_p = 1.0 - self.p
        mask = torch.bernoulli(torch.ones(shape, device=x.device, dtype=x.dtype, requires_grad=False), p=keep_p)
        # We don't rescale other batches. This class is intended to be used as LayerDrop.
        return mask
