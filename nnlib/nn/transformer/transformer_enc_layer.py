from typing import Optional, Tuple
import torch

from nnlib.nn.modules import BaseModule, LayerNorm
from nnlib.nn.transformer.multihead_attention import MultiheadSelfAttention
from nnlib.nn.transformer.projection import ProjectionResidualNorm, ProjectionResidual
from nnlib.nn.transformer.feed_forward import FeedForwardResidualNorm, NormFeedForwardResidual


class TransformerEncoderLayer(BaseModule):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 feedforward_dim: int = 2048,
                 attn_drop_prob: float = 0.1,
                 proj_drop_prob: float = 0.1,
                 qkv_drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None, *,
                 pre_norm: bool = False,
                 eps: float = 1e-5,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = MultiheadSelfAttention(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob,
                                           axial_drop=axial_drop)  # bias=True

        self.pre_norm = pre_norm
        if not pre_norm:
            self.norm = None
            self.projection = ProjectionResidualNorm(
                hidden_dim, hidden_dim, proj_drop_prob, eps=eps, axial_drop=axial_drop, norm_layer=norm_layer
            )
            self.feed_forward = FeedForwardResidualNorm(
                hidden_dim, feedforward_dim, proj_drop_prob, feedforward_drop_prob=feedforward_drop_prob, eps=eps,
                axial_drop=axial_drop, act_layer=act_layer, norm_layer=norm_layer
            )
        else:
            if norm_layer is None:
                norm_layer = LayerNorm
            self.norm = norm_layer(hidden_dim, eps=eps)
            self.projection = ProjectionResidual(
                hidden_dim, hidden_dim, proj_drop_prob, axial_drop=axial_drop
            )
            self.feed_forward = NormFeedForwardResidual(
                hidden_dim, feedforward_dim, proj_drop_prob, feedforward_drop_prob=feedforward_drop_prob, eps=eps,
                axial_drop=axial_drop, act_layer=act_layer, norm_layer=norm_layer
            )

    def forward(self,
                hidden: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, *,
                return_prob: bool = False) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]:
        """
        hidden:     (batch_size, query_length, hidden_dim)
        attn_mask:  (query_length, query_length) | (batch_size, query_length, query_length)
        """
        if not self.pre_norm:
            if return_prob:
                attn, prob = self.attn(hidden, attn_mask=attn_mask, return_prob=return_prob)
            else:
                attn = self.attn(hidden, attn_mask=attn_mask, return_prob=return_prob)
                prob = None  # fixed to always return tensors in output
            proj = self.projection(attn, hidden)
            output = self.feed_forward(proj)
        else:
            hidden_norm = self.norm(hidden)
            if return_prob:
                attn, prob = self.attn(hidden_norm, attn_mask=attn_mask, return_prob=return_prob)
            else:
                attn = self.attn(hidden_norm, attn_mask=attn_mask, return_prob=return_prob)
                prob = None
            proj = self.projection(attn, hidden)
            output = self.feed_forward(proj)
        return output, (prob,)
