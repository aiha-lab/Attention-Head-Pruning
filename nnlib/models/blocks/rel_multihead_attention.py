from typing import Optional, Union, Tuple
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from nnlib.nn import Linear, ParameterModule
from nnlib.nn.transformer.multihead_attention import _MultiheadAttentionBase
from nnlib.nn.transformer.utils import (apply_attn_mask, apply_weak_attention_suppression, bmm_4d, safe_softmax)


class RelativeMultiheadAttention(_MultiheadAttentionBase):

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
        super(RelativeMultiheadAttention, self).__init__(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob,
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
    def _rel_shift_left(x):
        """Shift relative length
        x: (batch_size, num_heads, query_length, key_length)

        6 5 4 3 2 1 0    4 3 2 1 0 - -
        6 5 4 3 2 1 0 -> 5 4 3 2 1 0 -
        6 5 4 3 2 1 0    6 5 4 3 2 1 0

        x[:, 0] << shift left (q - 1)
        x[:, 1] << shift left (q - 2)
        ...
        x[:, q-1] << shift left 0
        """
        batch_size, num_heads, query_len, key_len = x.shape
        x_padded = F.pad(x, (1, 0))

        x_padded = x_padded.view(batch_size, num_heads, 1 + key_len, query_len)  # (..., 1 + key_len, query_len)
        x = x_padded[:, :, 1:, :].view_as(x)  # (..., query_len, key_len)
        return x

    def forward(self,
                query_state: torch.Tensor,
                pos_emb: torch.Tensor,
                query_key_bias: Optional[torch.Tensor],
                query_pos_bias: Optional[torch.Tensor],
                memory: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, *,
                return_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        query_state:        (batch_size, query_length, hidden_dim)
        pos_emb:            (1, memory_length + query_length = key_length, hidden_dim), term B (R) in paper
        query_key_bias:     (hidden_dim,)   shared through all layers,              term C (u) in paper
        query_pos_bias:     (hidden_dim,)   shared through all layers,              term D (v) in paper
        memory:             (batch_size, memory_length, hidden_dim)

        attn_mask:          (query_length, key_length) | (batch_size, query_length, key_length)
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)

        if memory is not None:
            kv_query_state = torch.cat([memory, query_state], dim=1)  # (batch_size, memory_length + query_length, d)
        else:
            kv_query_state = query_state
        k = self.k_proj(kv_query_state)  # (batch_size, memory_len + query_len, hidden_dim)
        v = self.v_proj(kv_query_state)  # (batch_size, memory_len + query_len, hidden_dim)
        key_len = k.shape[1]

        if not self.use_axial_drop:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
            v = self.qkv_drop(v)
        else:  # axial drop, apply same mask for Q, K, V
            qkv_drop_mask = self.qkv_drop.generate_mask(k)
            k = self.qkv_drop(k, qkv_drop_mask)
            v = self.qkv_drop(v, qkv_drop_mask)
            q = self.qkv_drop(q, qkv_drop_mask[:, -query_len:, :])

        pos_emb = pos_emb[:, -key_len:]
        rk = self.r_proj(pos_emb)  # (1, memory_len + query_len, hidden_dim)
        if rk.shape[1] != key_len:
            raise ValueError(f"[ERROR:MODEL] Relative position sequence length {rk.shape[1]} mismatch, "
                             f"query_len: {query_len}, key_len: {key_len}.")
        rk = rk.expand(batch_size, rk.shape[1], hidden_dim)  # (batch_size, memory_len + query_len, hidden_dim)

        if self.share_query_bias:
            wq = q + query_key_bias  # (batch_size, query_len, hidden_dim)
            rq = q + query_pos_bias  # (batch_size, query_len, hidden_dim)
        else:
            wq = q + self.query_key_bias()
            rq = q + self.query_pos_bias()

        wq = self._transpose_for_attn(wq)  # (batch_size, num_heads, query_len, head_dim)
        wk = self._transpose_for_attn(k)  # (batch_size, num_heads, key_len, head_dim)
        rq = self._transpose_for_attn(rq)  # (batch_size, num_heads, query_len, head_dim)
        rk = self._transpose_for_attn(rk)  # (batch_size, num_heads, key_len, head_dim)

        v = self._transpose_for_attn(v)  # (batch_size, num_heads, key_len, head_dim)

        scores_ac = self.attn(wq, wk, mask=None)  # (batch_size, num_heads, query_len, key_len), no normalize
        scores_bd = self.attn(rq, rk, mask=None)  # (batch_size, num_heads, query_len, key_len), no normalize
        scores_bd = self._rel_shift_left(scores_bd)  # will be masked later, (batch_size, num_heads, query_len, key_len)

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
