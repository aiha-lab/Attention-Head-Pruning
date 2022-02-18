from typing import Optional, Tuple
import torch
import torch.nn as tnn
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.modules.linear import Linear
from nnlib.nn.parameter import ParameterModule


def _embedding_word_dropout(weight: torch.Tensor, training: bool, word_drop_prob: float) -> torch.Tensor:
    if training and (word_drop_prob > 0):
        num_embeddings = weight.shape[0]
        keep_p = 1.0 - word_drop_prob
        word_mask = torch.bernoulli(torch.ones(num_embeddings, 1, dtype=weight.dtype,
                                               device=weight.device), p=keep_p).bool()
        weight = (weight * word_mask).div(keep_p)
    return weight


class Embedding(BaseModule):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 *, word_drop_prob: float = 0.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= self.num_embeddings:
                    raise ValueError("[ERROR:NN] Embedding padding_idx must be within num_embeddings.")
            elif padding_idx < 0:
                if padding_idx < -self.num_embeddings:
                    raise ValueError("[ERROR:NN] Embedding padding_idx must be within num_embeddings")
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = ParameterModule(torch.empty(num_embeddings, embedding_dim).normal_(0, 1))
        self.sparse = sparse
        self.word_drop_prob = word_drop_prob

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        tnn.init.xavier_uniform_(self.weight.data)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight.data[self.padding_idx].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _embedding_word_dropout(self.weight(), self.training, self.word_drop_prob)
        return F.embedding(x,
                           weight,
                           self.padding_idx,
                           self.max_norm,
                           self.norm_type,
                           self.scale_grad_by_freq,
                           self.sparse)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.word_drop_prob > 0:
            s += f", word_drop_prob={self.word_drop_prob}"
        if self.scale_grad_by_freq is not False:
            s += f", scale_grad_by_freq={self.scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s


class EmbeddingBag(BaseModule):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 word_drop_prob: float = 0.0,
                 scale_grad_by_freq: bool = False,
                 mode: str = "mean",
                 sparse: bool = False,
                 include_last_offset: bool = False) -> None:
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode

        self.weight = ParameterModule(torch.empty(num_embeddings, embedding_dim).normal_(0, 1))
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.word_drop_prob = word_drop_prob

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        tnn.init.xavier_uniform_(self.weight.data)

    def forward(self, x: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = _embedding_word_dropout(self.weight(), self.training, self.word_drop_prob)
        return F.embedding_bag(x,
                               weight,
                               offsets,
                               self.max_norm,
                               self.norm_type,
                               self.scale_grad_by_freq,
                               self.mode,
                               self.sparse,
                               per_sample_weights=per_sample_weights,
                               include_last_offset=self.include_last_offset)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.word_drop_prob > 0:
            s += f", word_drop_prob={self.word_drop_prob}"
        if self.scale_grad_by_freq is not False:
            s += f", scale_grad_by_freq={self.scale_grad_by_freq}"
        if self.word_drop_prob > 0:
            s += f", word_drop_prob={self.word_drop_prob}"
        s += f", mode={self.mode}"
        return s


class LogSoftmaxWithLoss(BaseModule):

    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 bias: bool = True,
                 reduction: str = "mean") -> None:
        super(LogSoftmaxWithLoss, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.use_bias = bias
        self.reduction = reduction

        self.project = Linear(self.num_features, self.num_classes, bias=bias)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] != target.shape[0]:
            raise RuntimeError(f"[ERROR:NN] Target and Input should have same batch dimension. "
                               f"Shape of input: {x.shape}, target: {target.shape}")

        assert x.ndim == 2
        h = self.project(x)
        output = torch.log_softmax(h, dim=1)
        loss = F.nll_loss(output, target, reduction=self.reduction)
        return output, loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        h = self.project(x)
        pred = torch.argmax(h, dim=1)
        return pred
