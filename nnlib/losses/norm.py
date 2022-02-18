from typing import Dict, Any, Iterable, Union
import torch

from nnlib.losses import register_loss
from nnlib.losses.loss import BaseLoss


class LpLossToZero(BaseLoss):
    """Different to L1Loss / L2loss in pytorch, this class handles multiple tensor input."""

    def __init__(self, p=2):
        super(LpLossToZero, self).__init__(reduction="sum")
        self.p = p

    def forward(self, tensor_or_tensors: Union[torch.Tensor, Iterable[torch.Tensor]]) -> torch.Tensor:
        if isinstance(tensor_or_tensors, torch.Tensor):
            tensor_or_tensors = [tensor_or_tensors]
        loss = torch.zeros(1, dtype=tensor_or_tensors[0].dtype, device=tensor_or_tensors[0].device)
        for t in tensor_or_tensors:
            loss += torch.norm(t, p=self.p)
        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(p=config.get("p", 2.0))

    def extra_repr(self) -> str:
        return f"p={self.p}"


@register_loss("L1LossToZero")
class L1LossToZero(LpLossToZero):

    def __init__(self):
        super(L1LossToZero, self).__init__(p=1)


@register_loss("L2LossToZero")
class L2LossToZero(LpLossToZero):

    def __init__(self):
        super(L2LossToZero, self).__init__(p=2)
