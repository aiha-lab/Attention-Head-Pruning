from typing import Optional, Union, Tuple
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from nnlib.nn import Linear, ParameterModule
from nnlib.nn.transformer.multihead_attention import _MultiheadAttentionBase
from nnlib.nn.transformer.utils import (apply_attn_mask, apply_weak_attention_suppression, bmm_4d, safe_softmax)


class BidirectionalRelativeMultiheadAttention(_MultiheadAttentionBase):
    """Bidirectional Relative MHA.
    Unlike Transformer-XL, this module does not support "memory".
    Always all context should be input.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 qkv_drop_prob: float = 0.0,
                 *, axial_drop: bool = False,
                 share_query_bias: bool = True,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 was: bool = False,
                 was_gamma: float = 0.5):
        super(BidirectionalRelativeMultiheadAttention, self).__init__(hidden_dim, num_heads, attn_drop_prob,
                                                                      qkv_drop_prob,
                                                                      bias=False, attn_normalize=False,  # CRUCIAL!
                                                                      axial_drop=axial_drop, k_dim=k_dim, v_dim=v_dim)
        self.r_proj = Linear(hidden_dim, hidden_dim, bias=False)

        self.share_query_bias = share_query_bias
        if not share_query_bias:
            self.query_key_bias = ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
            self.query_pos_bias = ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
        else:
            self.query_key_bias = self.query_pos_bias = None

        self.was = was
        self.was_gamma = was_gamma

    @staticmethod
    def _rel_shift_right(x):
        """Shift relative length
        (THIS IS DIFFERENT TO TXL IMPLEMENTATION; NOT LEFT SHIFT!)

                                      |--use--|
        -3 -2 -1  0  1  2    -3 -2 -1  0  1  2
        -3 -2 -1  0  1  2 ->  - -3 -2 -1  0  1
        -3 -2 -1  0  1  2     -  - -3 -2 -1  0

        x: (batch_size, num_heads, query_length, key_length)
        x[:, 0] >> shift right 0
        x[:, 1] >> shift right 1
        ...
        x[:, q-1] >> shift right (q - 1)

        """
        batch_size, num_heads, query_len, key_len = x.shape
        x_padded = F.pad(x, (0, 1))

        x_padded = x_padded.view(batch_size, num_heads, -1)
        x = x_padded[:, :, :query_len * key_len].view_as(x)
        return x

    def forward(self,
                query_state: torch.Tensor,
                pos_emb: torch.Tensor,
                query_key_bias: Optional[torch.Tensor],
                query_pos_bias: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor] = None, *,
                return_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        query_state:        (batch_size, query_length, hidden_dim)
        pos_emb:            (1, query_length * 2, hidden_dim)       [-s, ... -2, -1, 0, 1, 2, 3, ... s-1]
        query_key_bias:     (hidden_dim,)    shared through all layers
        query_pos_bias:     (hidden_dim,)    shared through all layers

        attn_mask:          (batch_size, query_length, query_length)
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)
        k = self.k_proj(query_state)
        v = self.v_proj(query_state)

        if not self.use_axial_drop:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
            v = self.qkv_drop(v)
        else:  # axial drop, apply same mask for Q, K, V
            qkv_drop_mask = self.qkv_drop.generate_mask(k)
            q = self.qkv_drop(q, qkv_drop_mask)
            k = self.qkv_drop(k, qkv_drop_mask)
            v = self.qkv_drop(v, qkv_drop_mask)

        rk = self.r_proj(pos_emb)  # (1, query_len * 2, hidden_dim)
        rk = rk.expand(batch_size, -1, hidden_dim)

        if self.share_query_bias:
            wq = q + query_key_bias  # (batch_size, query_len, hidden_dim)
            rq = q + query_pos_bias  # (batch_size, query_len, hidden_dim)
        else:
            wq = q + self.query_key_bias()
            rq = q + self.query_pos_bias()

        wq = self._transpose_for_attn(wq)  # (batch_size, num_heads, query_len, head_dim)
        wk = self._transpose_for_attn(k)
        rq = self._transpose_for_attn(rq)
        rk = self._transpose_for_attn(rk)

        v = self._transpose_for_attn(v)  # (batch_size, num_heads, query_len, head_dim)

        scores_ac = self.attn(wq, wk, mask=None)  # (batch_size, num_heads, query_len, query_len), no normalize
        scores_bd = self.attn(rq, rk, mask=None)  # (batch_size, num_heads, query_len, query_len * 2)
        scores_bd = self._rel_shift_right(scores_bd)
        scores_bd = scores_bd[:, :, :, -query_len:]

        with amp.autocast(enabled=False):
            scores = scores_ac.float() + scores_bd.float()  # sum at un-normalized state
            scores = apply_attn_mask(scores, mask=attn_mask)
            if self.was:
                scores = apply_weak_attention_suppression(scores, gamma=self.was_gamma)
            # scores = torch.softmax(scores, dim=-1, dtype=torch.float32)
            scores = safe_softmax(scores, dim=-1, mask=attn_mask)
        scores = self.attn_drop(scores)

        # output = torch.einsum('bhqk,bhkd->bhqd', scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = bmm_4d(scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = self._transpose_for_output(output)  # (batch_size, query_len, hidden_dim)

        if return_prob:
            return output, scores
        else:
            return output
