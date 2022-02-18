from typing import Optional, Tuple, Dict
import torch

from nnlib.nn.modules import BaseModule, LayerNorm
from nnlib.nn.transformer.multihead_attention import MultiheadSelfAttention, MultiheadCrossAttention
from nnlib.nn.transformer.projection import ProjectionResidualNorm, ProjectionResidual
from nnlib.nn.transformer.feed_forward import FeedForwardResidualNorm, NormFeedForwardResidual


class TransformerDecoderLayer(BaseModule):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 feedforward_dim: int = 2048,
                 attn_drop_prob: float = 0.1,
                 proj_drop_prob: float = 0.1,
                 qkv_drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None, *,
                 cross_dim: Optional[int] = None,
                 pre_norm: bool = False,
                 eps: float = 1e-5,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(TransformerDecoderLayer, self).__init__()

        self.attn = MultiheadSelfAttention(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob,
                                           axial_drop=axial_drop)  # bias=True

        self.pre_norm = pre_norm
        if not pre_norm:
            self.norm = None
            self.proj_norm = None
            self.cross_norm = None
            self.projection = ProjectionResidualNorm(
                hidden_dim, hidden_dim, proj_drop_prob, eps=eps, axial_drop=axial_drop, norm_layer=norm_layer
            )
            self.cross_attn = MultiheadCrossAttention(
                hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob, axial_drop=axial_drop,
                k_dim=cross_dim, v_dim=cross_dim
            )
            self.cross_projection = ProjectionResidualNorm(
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
            self.proj_norm = norm_layer(hidden_dim, eps=eps)
            self.cross_norm = norm_layer(hidden_dim, eps=eps)
            self.projection = ProjectionResidual(
                hidden_dim, hidden_dim, proj_drop_prob, axial_drop=axial_drop
            )
            self.cross_attn = MultiheadCrossAttention(
                hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob, axial_drop=axial_drop,
                k_dim=cross_dim, v_dim=cross_dim
            )
            self.cross_projection = ProjectionResidual(
                hidden_dim, hidden_dim, proj_drop_prob, axial_drop=axial_drop
            )
            self.feed_forward = NormFeedForwardResidual(
                hidden_dim, feedforward_dim, proj_drop_prob, feedforward_drop_prob=feedforward_drop_prob, eps=eps,
                axial_drop=axial_drop, act_layer=act_layer, norm_layer=norm_layer
            )

    def forward(self,
                hidden: torch.Tensor,
                cross_hidden: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None, *,
                return_prob: bool = False) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]:
        """
        hidden:             (batch_size, query_length, hidden_dim)
        cross_hidden:       (batch_size, key_length, hidden_dim)
        attn_mask:          (query_length, query_length) | (batch_size, query_length, query_length)
        cross_attn_mask:    (query_length, key_length) | (batch_size, query_length, key_length)
        """
        if not self.pre_norm:
            if return_prob:
                attn, prob = self.attn(hidden, attn_mask=attn_mask, return_prob=return_prob)
            else:
                attn = self.attn(hidden, attn_mask=attn_mask, return_prob=return_prob)
                prob = None  # fixed to always return tensors in output
            proj = self.projection(attn, hidden)

            if return_prob:
                cross_attn, cross_prob = self.cross_attn(proj, cross_hidden, cross_hidden,
                                                         attn_mask=cross_attn_mask, return_prob=return_prob)
            else:
                cross_attn = self.cross_attn(proj, cross_hidden, cross_hidden,
                                             attn_mask=cross_attn_mask, return_prob=return_prob)
                cross_prob = None
            cross_proj = self.cross_projection(cross_attn, proj)
            output = self.feed_forward(cross_proj)
        else:
            hidden_norm = self.norm(hidden)
            if return_prob:
                attn, prob = self.attn(hidden_norm, attn_mask=attn_mask, return_prob=return_prob)
            else:
                attn = self.attn(hidden_norm, attn_mask=attn_mask, hreturn_prob=return_prob)
                prob = None
            proj = self.projection(attn, hidden)

            proj_norm = self.proj_norm(proj)
            cross_hidden_norm = self.cross_norm(cross_hidden)
            if return_prob:
                cross_attn, cross_prob = self.cross_attn(proj_norm, cross_hidden_norm, cross_hidden_norm,
                                                         attn_mask=cross_attn_mask, return_prob=return_prob)
            else:
                cross_attn = self.cross_attn(proj_norm, cross_hidden_norm, cross_hidden_norm,
                                             attn_mask=cross_attn_mask, return_prob=return_prob)
                cross_prob = None
            cross_proj = self.cross_projection(cross_attn, proj)
            output = self.feed_forward(cross_proj)

        return output, (prob, cross_prob)

    def step(self,
             hidden: torch.Tensor,
             cross_hidden: torch.Tensor,
             cross_attn_mask: Optional[torch.Tensor] = None, *,
             incremental_state: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        incremental_state:      {'prev_key', 'prev_value', 'cross_key', 'cross_value'}
        """

        if not self.pre_norm:
            attn = self.attn.step(hidden, prev_state=incremental_state)
            proj = self.projection(attn, hidden)
            cross_attn = self.cross_attn.step(proj, cross_hidden, cross_hidden,
                                              attn_mask=cross_attn_mask, cross_state=incremental_state)
            cross_proj = self.cross_projection(cross_attn, proj)
            output = self.feed_forward(cross_proj)
        else:
            hidden_norm = self.norm(hidden)
            attn = self.attn.step(hidden_norm, prev_state=incremental_state)
            proj = self.projection(attn, hidden)

            proj_norm = self.proj_norm(proj)
            cross_hidden_norm = self.cross_norm(cross_hidden)
            cross_attn = self.cross_attn.step(proj_norm, cross_hidden_norm, cross_hidden_norm,
                                              attn_mask=cross_attn_mask, cross_state=incremental_state)
            cross_proj = self.cross_projection(cross_attn, proj)
            output = self.feed_forward(cross_proj)

        return output
