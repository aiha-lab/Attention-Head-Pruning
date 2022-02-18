from typing import Optional, Tuple
import torch

import nnlib.nn as nn

from all_transformerl_xl_infer.rel_multihead_all_attention_infer import RelativeMultiheadAllAttentionInfer
from nnlib.nn.transformer.projection import ProjectionResidualNorm, ProjectionResidual


class RelativeAllTransformerInferLayer(nn.BaseModule):

    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int,
                 attn_bias: bool = True, share_r_bias: bool = True,
                 pre_norm: bool = True, *, effective_heads: Optional[int] = None,
                 name: Optional[str] = None, norm_layer=None):
        super(RelativeAllTransformerInferLayer, self).__init__()

        # feedforward dim as memory length
        self.attn = RelativeMultiheadAllAttentionInfer(hidden_dim, num_heads, feedforward_dim,
                                                       effective_heads=effective_heads,
                                                       bias=attn_bias, share_r_bias=share_r_bias)
        effective_dim = self.attn.effective_dim

        self.pre_norm = pre_norm
        if not pre_norm:
            self.norm = None
            self.projection = ProjectionResidualNorm(effective_dim, hidden_dim, norm_layer=norm_layer)
        else:
            if norm_layer is None:
                norm_layer = nn.LayerNorm
            self.norm = norm_layer(hidden_dim)
            self.projection = ProjectionResidual(effective_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor,
                     rel_pos: torch.Tensor,
                     rel_query_bias: Optional[torch.Tensor], rel_pos_bias: Optional[torch.Tensor],
                     memory: Optional[torch.Tensor] = None,
                     *, attn_mask: Optional[torch.Tensor] = None,
                     head_mask: Optional[torch.Tensor] = None,
                     return_prob: bool = False) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]:
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
            attn, prob = self.attn(hidden, rel_pos, rel_query_bias, rel_pos_bias, memory=memory,
                                   attn_mask=attn_mask, head_mask=head_mask, return_prob=return_prob)
            output = self.projection(attn, hidden)
        else:
            hidden_norm = self.norm(hidden)
            if memory is not None:
                memory_norm = self.norm(memory)
            else:
                memory_norm = memory
            attn, prob = self.attn(hidden_norm, rel_pos, rel_query_bias, rel_pos_bias,
                                   memory=memory_norm, attn_mask=attn_mask, head_mask=head_mask,
                                   return_prob=return_prob)
            output = self.projection(attn, hidden)
        return output, (prob,)


if __name__ == '__main__':
    from happy_torch.models.transformer_xl.rel_transformer_layer import RelativeTransformerLayer

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
