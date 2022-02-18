import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule


class PairwiseDistance(BaseModule):

    def __init__(self,
                 p: float = 2.0,
                 eps: float = 1e-6,
                 keepdim: bool = False) -> None:
        super(PairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.pairwise_distance(x1, x2, self.norm, self.eps, self.keepdim)


class CosineSimilarity(BaseModule):

    def __init__(self,
                 dim: int = 1,
                 eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
