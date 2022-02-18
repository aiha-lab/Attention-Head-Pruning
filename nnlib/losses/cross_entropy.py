from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from nnlib.losses import register_loss
from nnlib.losses.loss import BaseLoss


@register_loss("CELossWithSmoothing")
class CELossWithSmoothing(BaseLoss):

    def __init__(self,
                 smoothing: float = 0,
                 ignore_index: int = -100,
                 *, reduction: str = "mean"):
        super(CELossWithSmoothing, self).__init__(reduction=reduction)
        if not (0 <= smoothing < 1):
            raise ValueError(f"[ERROR:LOSS] Invalid label smoothing: {smoothing}, should be in range [0, 1)")
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                *, smoothing: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (loss_with_smoothing, loss_without_smoothing)"""
        if smoothing is None:  # override
            smoothing = self.smoothing

        output = output.float()

        if smoothing == 0:
            loss = F.cross_entropy(output, target, ignore_index=self.ignore_index, reduction=self.reduction)
            return loss, loss
        else:  # label smoothing
            log_prob = F.log_softmax(output, dim=-1)  # (n, c)
            num_classes = log_prob.shape[-1]

            nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(-1))  # (n, 1)
            smooth_loss = -log_prob.sum(dim=-1, keepdim=True)  # (n, 1)
            if self.ignore_index >= 0:
                ignore_mask = target.eq(self.ignore_index)
                if ignore_mask.ndim == 1:  # temporal expand
                    ignore_mask = ignore_mask.view(-1, 1)
                nll_loss.masked_fill_(ignore_mask, 0.0)
                smooth_loss.masked_fill_(ignore_mask, 0.0)

            nll_loss = nll_loss.squeeze()  # (n,)
            smooth_loss = smooth_loss.squeeze()  # (n,)

            smoothing_val = smoothing / (num_classes - 1)
            loss = (1.0 - smoothing - smoothing_val) * nll_loss + smoothing_val * smooth_loss

            if self.reduction == "sum":
                return loss.sum(), nll_loss.sum()
            elif self.reduction == "mean":
                return loss.mean(), nll_loss.mean()
            else:  # none
                return loss, nll_loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(
            smoothing=config.get("smoothing", 0.0),
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean"),
        )

    def extra_repr(self) -> str:
        return f"smoothing={self.smoothing}, ignore_index={self.ignore_index}"
