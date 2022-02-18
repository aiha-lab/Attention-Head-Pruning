from typing import Optional, Tuple
import torch

import nnlib.nn as nn

from all_transformer_xl.rel_multihead_all_attention import RelativeMultiheadAllAttention
from nnlib.nn.transformer.projection import ProjectionResidualNorm, ProjectionResidual


class RelativeAllTransformerLayer(nn.BaseModule):

    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int,
                 attn_drop_prob: float = 0.1, proj_drop_prob: float = 0.1, qkv_drop_prob: float = 0.0,
                 attn_bias: bool = True, share_r_bias: bool = True,
                 pre_norm: bool = True, *, axial_drop: bool = False, name: Optional[str] = None,
                 norm_layer=None):
        super(RelativeAllTransformerLayer, self).__init__()

        # feedforward dim as memory length
        self.attn = RelativeMultiheadAllAttention(hidden_dim, num_heads, feedforward_dim, attn_drop_prob, qkv_drop_prob,
                                                  axial_drop=axial_drop,
                                                  bias=attn_bias, share_r_bias=share_r_bias)

        self.pre_norm = pre_norm
        if not pre_norm:
            self.norm = None
            self.projection = ProjectionResidualNorm(hidden_dim, hidden_dim, proj_drop_prob,
                                                     axial_drop=axial_drop, norm_layer=norm_layer)
        else:
            if norm_layer is None:
                norm_layer = nn.LayerNorm
            self.norm = norm_layer(hidden_dim)
            self.projection = ProjectionResidual(hidden_dim, hidden_dim, proj_drop_prob, axial_drop=axial_drop)

    def set_gating(self, flag: bool = True):
        self.attn.set_gating(flag)

    def forward(self, hidden: torch.Tensor,
                     rel_pos: torch.Tensor,
                     rel_query_bias: Optional[torch.Tensor], rel_pos_bias: Optional[torch.Tensor],
                     memory: Optional[torch.Tensor] = None,
                     *, attn_mask: Optional[torch.Tensor] = None,
                     head_mask: Optional[torch.Tensor] = None,
                     return_prob: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                         Tuple[torch.Tensor, int],
                                                         Tuple[Optional[torch.Tensor], ...]]:
        """
        hidden:         (batch_size, target_length, hidden_dim)
        rel_pos:        (1, target_length + memory_length, hidden_dim)
        rel_query_bias: (hidden_dim,)
        rel_pos_bias:   (hidden_dim,)
        memory:         (batch_size, memory_length, hidden_dim)

        attn_mask:  (target_length, source_length) | (batch_size, target_length, source_length)
        head_mask:  (num_heads) | (batch_size, num_heads)
        """

        if not self.pre_norm:
            attn, loss, sparsity, prob = self.attn(hidden, rel_pos, rel_query_bias, rel_pos_bias, memory=memory,
                                                   attn_mask=attn_mask, head_mask=head_mask, return_prob=return_prob)
            output = self.projection(attn, hidden)
        else:
            hidden_norm = self.norm(hidden)
            if memory is not None:
                memory_norm = self.norm(memory)
            else:
                memory_norm = memory
            attn, loss, sparsity, prob = self.attn(hidden_norm, rel_pos, rel_query_bias, rel_pos_bias,
                                                   memory=memory_norm, attn_mask=attn_mask, head_mask=head_mask,
                                                   return_prob=return_prob)
            output = self.projection(attn, hidden)
        return output, loss, sparsity, (prob,)


if __name__ == '__main__':
    from nnlib.models.transformer_xl.rel_transformer_layer import RelativeTransformerLayer

    m1 = RelativeTransformerLayer(512, 8, 2048, pre_norm=True)
    m2 = RelativeAllTransformerLayer(512, 8, 2048, pre_norm=True)

    m1_count = 0
    for p_n, p in m1.named_parameters():
        print(p_n, p.shape)
        m1_count += p.numel()

    m2_count = 0
    for p_n, p in m2.named_parameters():
        print(p_n, p.shape)
        m2_count += p.numel()

    print("M1", m1_count, "M2", m2_count)
