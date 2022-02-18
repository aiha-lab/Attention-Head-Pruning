from typing import Optional, Union, Tuple, Dict
import torch

from nnlib.nn.modules import BaseModule, Linear, Dropout, SequenceDropout
from nnlib.nn.transformer.dot_product import ScaledDotProduct
from nnlib.nn.transformer.utils import bmm_4d


class _MultiheadAttentionBase(BaseModule):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 qkv_drop_prob: float = 0.0,
                 bias: bool = True, *,
                 attn_normalize: bool = True,
                 axial_drop: bool = False,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 was: bool = False,
                 was_gamma: float = 0.5):
        super(_MultiheadAttentionBase, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = bias
        if num_heads <= 0:
            raise ValueError(f"[ERROR:NN] Number of heads should be positive, but got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"[ERROR:NN] Hidden dim {hidden_dim} should be multiple of num_heads {num_heads}.")
        self.head_dim = hidden_dim // num_heads

        if k_dim is None:
            k_dim = hidden_dim
        if v_dim is None:
            v_dim = k_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.q_proj = Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = Linear(k_dim, hidden_dim, bias=bias)
        self.v_proj = Linear(v_dim, hidden_dim, bias=bias)

        self.use_axial_drop = axial_drop
        self.qkv_drop = SequenceDropout(qkv_drop_prob, axial_drop=axial_drop, axis=1, inplace=True)  # seq axis

        self.attn = ScaledDotProduct(self.head_dim, normalize=attn_normalize,
                                     was=was, was_gamma=was_gamma)
        self.attn_drop = Dropout(attn_drop_prob, inplace=False)

    def _transpose_for_attn(self, x: torch.Tensor):
        # (batch_size, length, hidden_dim) -> (batch_size, num_heads, length, head_dim)
        batch_size, length, hidden_dim = x.shape
        head_dim = hidden_dim // self.num_heads
        x = x.view(batch_size, length, self.num_heads, head_dim).transpose(1, 2).contiguous()
        return x

    def _transpose_for_output(self, x: torch.Tensor):
        # (batch_size, num_heads, length, head_dim) -> (batch_size, length, hidden_dim)
        batch_size, num_heads, length, head_dim = x.shape
        x = x.transpose(2, 1).contiguous().view(batch_size, length, self.hidden_dim)
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = f"{self.hidden_dim}, num_heads={self.num_heads}"
        if self.attn_drop.p > 0:
            s += f", attn_drop_prob={self.attn_drop.p}"
        if self.qkv_drop.p > 0:
            s += f", qkv_drop_prob={self.qkv_drop.p}"
            if self.use_axial_drop:
                s += f", axial_drop=True"
        if not self.use_bias:
            s += f", bias=False"
        if self.k_dim != self.hidden_dim:
            s += f", k_dim={self.k_dim}"
        if self.v_dim != self.hidden_dim:
            s += f", v_dim={self.v_dim}"
        return s


class MultiheadSelfAttention(_MultiheadAttentionBase):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 qkv_drop_prob: float = 0.0,
                 bias: bool = True, *,
                 axial_drop: bool = False,
                 was: bool = False,
                 was_gamma: float = 0.5):
        super(MultiheadSelfAttention, self).__init__(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob, bias,
                                                     attn_normalize=True, axial_drop=axial_drop,
                                                     was=was, was_gamma=was_gamma)

    def forward(self,
                query_state: torch.Tensor, *,
                attn_mask: Optional[torch.Tensor] = None,
                return_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        query_state:    (batch_size, query_length, hidden_dim)
        attn_mask:      (query_length, query_length) | (batch_size * num_heads, query_length, query_length)
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)
        k = self.k_proj(query_state)  # (batch_size, query_len, hidden_dim)
        v = self.v_proj(query_state)  # (batch_size, query_len, hidden_dim)

        if not self.use_axial_drop:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
            v = self.qkv_drop(v)
        else:  # axial drop. apply for all Q, K, V
            q_drop_mask = self.qkv_drop.generate_mask(q)
            q = self.qkv_drop(q, q_drop_mask)
            k = self.qkv_drop(k, q_drop_mask)
            v = self.qkv_drop(v, q_drop_mask)

        q = self._transpose_for_attn(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._transpose_for_attn(k)  # (batch_size, num_heads, query_len, head_dim)
        v = self._transpose_for_attn(v)  # (batch_size, num_heads, query_len, head_dim)

        scores = self.attn(q, k, mask=attn_mask)  # (batch_size, num_heads, query_len, query_len)
        scores = self.attn_drop(scores)

        # output = torch.einsum("bhqk,bhkd->bhqd", scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = bmm_4d(scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = self._transpose_for_output(output)  # (batch_size, query_length, hidden_dim)

        if return_prob:
            return output, scores
        else:
            return output

    def step(self,
             query_state: torch.Tensor,
             prev_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """For generation, query_state is expected to get seq_len == 1 (not restricted). Assume eval() mode.
        query_state:
        prev_state:     {'prev_key', 'prev_value'}
            prev_key:       (batch_size, num_heads, prev_key_length, head_dim)
            prev_value:     (batch_size, num_heads, prev_key_length, head_dim)
            (1) will be automatically UPDATED after run
            (2) will attend ALL
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        # load previous states
        prev_key = prev_value = None
        prev_len = 0
        if len(prev_state) > 0:
            prev_key = prev_state.get("prev_key")
            prev_value = prev_state.get("prev_value")
            if (prev_key is not None) and (prev_value is not None):
                _, _, prev_len, _ = prev_key.shape
                assert prev_key.shape == prev_value.shape == (batch_size, self.num_heads, prev_len, self.head_dim)

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)
        k = self.k_proj(query_state)  # (batch_size, query_len, hidden_dim)
        v = self.v_proj(query_state)  # (batch_size, query_len, hidden_dim)

        q = self._transpose_for_attn(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._transpose_for_attn(k)  # (batch_size, num_heads, query_len, head_dim)
        v = self._transpose_for_attn(v)  # (batch_size, num_heads, query_len, head_dim)

        if prev_len > 0:
            k = torch.cat([prev_key, k], dim=2)  # (batch_size, num_heads, query_len + prev_len, head_dim)
            v = torch.cat([prev_value, v], dim=2)  # (batch_size, num_heads, query_len + prev_len, head_dim)
        prev_state.update({"prev_key": k.detach().clone(), "prev_value": v.detach().clone()})

        scores = self.attn(q, k, mask=None)  # (batch_size, num_heads, query_len, query_len + prev_len)

        # output = torch.einsum("bhqk,bhkd->bhqd", scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = bmm_4d(scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = self._transpose_for_output(output)  # (batch_size, query_length, hidden_dim)
        return output


class MultiheadCrossAttention(_MultiheadAttentionBase):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 qkv_drop_prob: float = 0.0,
                 bias: bool = True, *,
                 axial_drop: bool = False,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 was: bool = False,
                 was_gamma: float = 0.5):
        super(MultiheadCrossAttention, self).__init__(hidden_dim, num_heads, attn_drop_prob, qkv_drop_prob, bias,
                                                      attn_normalize=True, axial_drop=axial_drop,
                                                      k_dim=k_dim, v_dim=v_dim, was=was, was_gamma=was_gamma)

    def forward(self,
                query_state: torch.Tensor,
                key_state: torch.Tensor,
                value_state: Optional[torch.Tensor] = None, *,
                attn_mask: Optional[torch.Tensor] = None,
                return_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        query_state:    (batch_size, query_length, hidden_dim)
        key_state:      (batch_size, key_length, hidden_dim)
        value_state:    (batch_size, key_length, hidden_dim)
        attn_mask:      (query_length, key_length) | (batch_size, query_length, key_length)
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        _, key_len, _ = key_state.shape
        if value_state is None:
            value_state = key_state
        assert value_state.shape[1] == key_len

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)
        k = self.k_proj(key_state)  # (batch_size, key_len, hidden_dim)
        v = self.v_proj(value_state)  # (batch_size, key_len, hidden_dim)

        if not self.use_axial_drop:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
            v = self.qkv_drop(v)
        else:  # axial drop. share mask for K, V
            q = self.qkv_drop(q)
            kv_drop_mask = self.qkv_drop.generate_mask(k)
            k = self.qkv_drop(k, kv_drop_mask)
            v = self.qkv_drop(v, kv_drop_mask)

        q = self._transpose_for_attn(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._transpose_for_attn(k)  # (batch_size, num_heads, key_len, head_dim)
        v = self._transpose_for_attn(v)  # (batch_size, num_heads, key_len, head_dim)

        scores = self.attn(q, k, mask=attn_mask)  # (batch_size, num_heads, query_len, key_len)
        scores = self.attn_drop(scores)

        # output = torch.einsum("bhqk,bhkd->bhqd", scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = bmm_4d(scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = self._transpose_for_output(output)  # (batch_size, query_length, hidden_dim)

        if return_prob:
            return output, scores
        else:
            return output

    def step(self,
             query_state: torch.Tensor,
             key_state: Optional[torch.Tensor] = None,
             value_state: Optional[torch.Tensor] = None, *,
             attn_mask: Optional[torch.Tensor] = None,
             cross_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """For generation, query_state is expected to get seq_len == 1 (not restricted). Assume eval() mode.
        query_state:
        cross_state:     {'cross_key', 'cross_value'}
            cross_key:       (batch_size, num_heads, cross_length, head_dim)
            cross_value:     (batch_size, num_heads, cross_length, head_dim)
            (1) if prev_state is given, ignore key_state and value_state.
            (2) if prev_state is empty, first create key and value, save.
        """
        batch_size, query_len, hidden_dim = query_state.shape
        assert hidden_dim == self.hidden_dim

        if value_state is None:
            value_state = key_state

        # load previous states
        cross_key = cross_value = None
        cross_len = 0
        if len(cross_state) > 0:
            cross_key = cross_state.get("cross_key")
            cross_value = cross_state.get("cross_value")
            if (cross_key is not None) and (cross_value is not None):
                _, _, cross_len, _ = cross_key.shape
                assert cross_key.shape == cross_value.shape == (batch_size, self.num_heads, cross_len, self.head_dim)

        q = self.q_proj(query_state)  # (batch_size, query_len, hidden_dim)
        q = self._transpose_for_attn(q)  # (batch_size, num_heads, query_len, head_dim)

        if cross_len == 0:  # should creat first one.
            assert (key_state is not None) and (value_state is not None)
            k = self.k_proj(key_state)  # (batch_size, cross_len, hidden_dim)
            v = self.v_proj(value_state)  # (batch_size, cross_len, hidden_dim)

            k = self._transpose_for_attn(k)  # (batch_size, num_heads, cross_len, head_dim)
            v = self._transpose_for_attn(v)  # (batch_size, num_heads, cross_len, head_dim)
            cross_state.update({"cross_key": k.detach().clone(), "cross_value": v.detach().clone()})
        else:
            assert (cross_key is not None) and (cross_value is not None)
            k = cross_key
            v = cross_value
            # do not update cross_state. already computed!

        scores = self.attn(q, k, mask=attn_mask)  # (batch_size, num_heads, query_len, cross_len)

        # output = torch.einsum("bhqk,bhkd->bhqd", scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = bmm_4d(scores, v)  # (batch_size, num_heads, query_len, head_dim)
        output = self._transpose_for_output(output)  # (batch_size, query_length, hidden_dim)
        return output
