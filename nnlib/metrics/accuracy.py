from typing import Tuple, Union, Dict, Any
import torch

from nnlib.metrics import register_metric
from nnlib.metrics.metric import BaseMetric


@register_metric("Accuracy")
class Accuracy(BaseMetric):

    def __init__(self, top_k: Union[int, Tuple[int, ...]] = 1) -> None:
        super(Accuracy, self).__init__()
        if isinstance(top_k, int):
            top_k = (top_k,)
        else:
            top_k = sorted(top_k)
        self.top_k = top_k

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size = target.shape[0]
        with torch.no_grad():
            if self.top_k == (1,):
                pred = torch.argmax(output, dim=1, keepdim=False)  # (n,)
                correct = torch.eq(pred, target).float().sum().div_(batch_size)  # (n,) -> (1,)
                return correct

            max_k = max(self.top_k)
            _, pred = torch.topk(output, max_k, dim=1, largest=True, sorted=True)  # (n, k)
            pred = pred.t()  # (n, k) -> (k, n)
            correct = torch.eq(pred, target.view(1, -1).expand_as(pred))  # (k, n)

            res = []
            for k in self.top_k:
                correct_k = correct[:k].reshape(-1).float().sum().div_(batch_size)  # (k, n) - > (kn,)
                res.append(correct_k)
            return torch.as_tensor(res, device=pred.device)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Accuracy":
        return cls(
            top_k=config.get("top_k", 1)
        )
