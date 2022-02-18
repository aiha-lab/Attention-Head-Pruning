from typing import Optional, Tuple
import torch
import math
import torch.nn as tnn
import torch.nn.functional as F

import nnlib.nn as nn
from nnlib.nn.transformer.multihead_attention import _MultiheadAttentionBase
from nnlib.nn.transformer.utils import apply_attn_mask

from .simple_gating import SimpleGating


class RelativeMultiheadAllAttention(_MultiheadAttentionBase):

    def __init__(self, hidden_dim: int, num_heads: int, num_persistent: int, attn_drop_prob: float = 0.0,
                 qkv_drop_prob: float = 0.0,
                 bias: bool = True, share_r_bias: bool = True, *,
                 axial_drop: bool = False,
                 k_dim: Optional[int] = None, v_dim: Optional[int] = None, name: Optional[str] = None):
        super(RelativeMultiheadAllAttention, self).__init__(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob, bias,
                                                            attn_normalize=False,  # CRUCIAL!
                                                            axial_drop=axial_drop,
                                                            k_dim=k_dim, v_dim=v_dim)
        self.r_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.share_r_bias = share_r_bias
        if not share_r_bias:
            self.r_query_bias = nn.ParameterModule(torch.zeros(hidden_dim, ))
            self.r_pos_bias = nn.ParameterModule(torch.zeros(hidden_dim, ))
        else:
            self.r_query_bias = self.r_pos_bias = None

        self.num_persistent = num_persistent
        self.mem_k = nn.ParameterModule(torch.zeros(1, self.num_heads, num_persistent, self.head_dim))
        self.mem_v = nn.ParameterModule(torch.zeros(1, self.num_heads, num_persistent, self.head_dim))

        s = math.sqrt(1 / self.hidden_dim)
        tnn.init.uniform_(self.mem_k.data, -s, s)
        tnn.init.normal_(self.mem_v.data, -s, s)

        # self.gate = SimpleGating(self.num_heads)
        # self.gate = SimpleGating(self.num_heads, init=0.0, beta=0.333, gamma=-0.1, zeta=1.1, hard=True)
        self.gate = SimpleGating(self.num_heads, init=2.0, beta=0.333, gamma=-0.1, zeta=1.1, hard=True)
        self.use_gate = False

    def set_gating(self, flag: bool = True):
        self.use_gate = flag

    def _rel_shift(self, x):
        """Shift relative length
        x: (batch_size, target_length, source_length (+ target_length))
        x[:, 0] << shift left (t - 1)
        x[:, 1] << shift left (t - 2)
        ...
        x[:, t-1] << shift left 0
        """
        batch_size, target_len, source_len = x.shape

        # zero_pad = torch.zeros(batch_size, target_len, 1, device=x.device, dtype=x.dtype)
        # x_padded = torch.cat([zero_pad, x], dim=-1)  # (batch_size, target_len, 1 + source_len)
        x_padded = F.pad(x, (1, 0))

        x_padded = x_padded.view(batch_size, 1 + source_len, target_len)  # (batch_size, 1 + source_len, target_len)
        x = x_padded[:, 1:, :].view_as(x)  # (batch_size, target_len, source_len)
        return x

    def _transpose_for_persistent(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.expand(batch_size, self.num_heads, self.num_persistent, self.head_dim).contiguous().view(
            batch_size * self.num_heads, self.num_persistent, self.head_dim)
        return x

    def forward(self, query_state: torch.Tensor,
                     rel_pos: torch.Tensor,
                     rel_query_bias: Optional[torch.Tensor],
                     rel_pos_bias: Optional[torch.Tensor],
                     memory: Optional[torch.Tensor] = None,
                     *, attn_mask: Optional[torch.Tensor] = None,
                     head_mask: Optional[torch.Tensor] = None,
                     return_prob: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                         Tuple[torch.Tensor, int], Optional[torch.Tensor]]:
        """
        query_state:    (batch_size, target_length, hidden_dim)
            XL does not have key_state nor value_state

        # rel_pos:            (1, memory_length + target_length = source_length, hidden_dim), term B in paper
        rel_pos:            (1, memory_length + target_length = source_length, head_dim), term B in paper
        rel_query_bias:     (hidden_dim,)       shared through all layers,                  term C in paper
        rel_pos_bias:       (hidden_dim,)       shared through all layers,                  term D in paper
        memory:             (batch_size, memory_length, hidden_dim)

        attn_mask:      (target_length, source_length)  | (batch_size, target_length, source_length)
        head_mask:      (num_heads) | (batch_size, num_heads)
        """
        batch_size, target_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        q = self.q_proj(query_state)  # (batch_size, target_len, hidden_dim)

        if memory is not None:
            kv_query_state = torch.cat([memory, query_state], dim=1)  # (batch_size, memory_length + target_length, d)
        else:
            kv_query_state = query_state
        k = self.k_proj(kv_query_state)  # (batch_size, memory_len + target_len, hidden_dim)
        v = self.v_proj(kv_query_state)  # (batch_size, memory_len + target_len, hidden_dim)
        source_len = k.shape[1]

        # TODO add qkv_drop

        rk = self.r_proj(rel_pos)  # (1, memory_len + target_len (* 2), hidden_dim)
        rk = rk[:, -source_len:, :]
        if rk.shape[1] != source_len:
            raise ValueError(f'Relative position sequence length {rk.shape[1]} mismatch, '
                             f'target_len: {target_len}, source_len: {source_len}.')
        # (batch_size, memory_len + target_len (* 2), hidden_dim)
        rk = rk.expand(batch_size, rk.shape[1], hidden_dim)

        if self.share_r_bias:
            if (rel_pos_bias is None) or (rel_query_bias is None):
                raise ValueError(f'Sharing relative bias, but given rel_pos_bias or rel_query_bias is None.')
        else:
            if (rel_pos_bias is not None) or (rel_query_bias is not None):
                raise ValueError(f'Not sharing relative bias, but given rel_pos_bias or rel_query_bias is not None.')
            rel_query_bias = self.r_query_bias()
            rel_pos_bias = self.r_pos_bias()

        wq = q + rel_query_bias  # (batch_size, target_len, hidden_dim)
        rq = q + rel_pos_bias  # (batch_size, target_len, hidden_dim)

        wq = self._transpose_for_attn(wq)  # (batch_size * num_heads, target_len, head_dim)
        wk = self._transpose_for_attn(k)  # (batch_size * num_heads, source_len, head_dim)
        rq = self._transpose_for_attn(rq)  # (batch_size * num_heads, target_len, head_dim)
        rk = self._transpose_for_attn(rk)  # (batch_size * num_heads, source_len (+ target_len), head_dim)
        # rk = rel_pos.expand(batch_size * self.num_heads, *rel_pos.shape[1:])

        v = self._transpose_for_attn(v)  # (batch_size * num_heads, source_len, head_dim)

        scores_ac = self.attn(wq, wk, mask=None)  # (batch_size * num_heads, target_len, source_len), no normalize
        scores_bd = self.attn(rq, rk, mask=None)  # (batch_size * num_heads, target_len, source_len (+ target_len))
        scores_bd = self._rel_shift(scores_bd)  # will be masked later, (batch_size * num_heads, target_len, source_len)

        scores = scores_ac + scores_bd  # sum at un-normalized state
        scores = apply_attn_mask(scores, mask=attn_mask)

        mem_k = self._transpose_for_persistent(self.mem_k(), batch_size)
        mem_k = mem_k * math.sqrt(self.head_dim)
        q = self._transpose_for_attn(q)
        scores_mem_k = self.attn(q, mem_k, mask=None)  # (batch_size * num_heads, target_len, persistent_len)

        # (batch_size * num_heads, target_len, source_len + persistent_len)
        scores = torch.cat([scores, scores_mem_k], dim=-1)

        scores = torch.softmax(scores, dim=-1)
        scores = self.attn_drop(scores)

        # (batch_size * num_heads, persistent_len, head_dim)
        mem_v = self._transpose_for_persistent(self.mem_v(), batch_size)
        mem_v = mem_v * math.sqrt(self.num_persistent)

        # (batch_size * num_heads, source_len + persistent_len, head_dim)
        v = torch.cat([v, mem_v], dim=1)

        output = torch.bmm(scores, v)  # (batch_size * num_heads, target_len, head_dim)

        gate, gate_loss, gate_sparsity = self.gate()
        if self.use_gate:
            gate = gate.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1, 1, 1)
            # gate_scale = torch.clamp(self.num_heads / (self.num_heads - gate_sparsity[0] + 1e-3),
            #                          1.0, self.num_heads).detach()
            # output = output * gate * gate_scale
            output = output * gate

        output = self._transpose_for_output(output)  # (batch_size, target_len, hidden_dim)

        if return_prob:
            scores_view = scores.view(batch_size, self.num_heads, target_len, source_len)
            return output, gate_loss, gate_sparsity, scores_view
        else:
            return output, gate_loss, gate_sparsity, None
