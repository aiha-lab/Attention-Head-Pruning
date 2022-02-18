from typing import Optional, Tuple
import math
import torch
import torch.nn as tnn
import torch.nn.functional as F

from nnlib.nn.modules import BaseModule, Linear
from nnlib.nn.parameter import ParameterModule


class TorchMultiheadAttention(BaseModule):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 drop_prob: float = 0.0,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: bool = False,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None):
        super(TorchMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if (kdim is not None) else embed_dim
        self.vdim = vdim if (vdim is not None) else embed_dim
        self._qkv_same_embed_dim = (self.kdim == embed_dim) and (self.vdim == embed_dim)
        self.add_zero_attn = add_zero_attn
        self.use_bias = bias

        self.num_heads = num_heads
        self.drop_prob = drop_prob
        if num_heads <= 0:
            raise ValueError(f"[ERROR:NN] Number of heads should be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"[ERROR:NN] Embed dimension is not multiple of num_heads.")
        self.head_dim = embed_dim // num_heads

        if self._qkv_same_embed_dim:
            # multiply at once
            self.in_proj = Linear(embed_dim, 3 * embed_dim, bias=bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            # separate multiply
            self.in_proj = None
            self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = Linear(self.kdim, embed_dim, bias=bias)
            self.v_proj = Linear(self.vdim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = ParameterModule(torch.zeros(embed_dim, ))
            self.bias_v = ParameterModule(torch.zeros(embed_dim, ))
        else:
            self.bias_k = self.bias_v = None

        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        if self._qkv_same_embed_dim:
            tnn.init.xavier_uniform_(self.in_proj.weight.data)
        else:
            tnn.init.xavier_uniform_(self.q_proj.weight.data)
            tnn.init.xavier_uniform_(self.k_proj.weight.data)
            tnn.init.xavier_uniform_(self.v_proj.weight.data)

        if self.in_proj.bias is not None:
            tnn.init.zeros_(self.in_proj.bias.data)
        tnn.init.zeros_(self.out_proj.bias.data)

        if self.bias_k is not None:
            tnn.init.normal_(self.bias_k.data, 0, math.sqrt(1.0 / self.embed_dim))
        if self.bias_v is not None:
            tnn.init.normal_(self.bias_v.data, 0, math.sqrt(1.0 / self.embed_dim))

    def attn_core(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  key_padding_mask: Optional[torch.Tensor] = None,
                  need_weights: bool = True,
                  attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        target_len, batch_size, embed_dim = query.shape
        if embed_dim != self.embed_dim:
            raise ValueError("[ERROR:NN] Input dimension mismatch")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("[ERROR:NN] Key or Value seq_length/batch_size mismatch")
        key_len, _, _ = key.shape  # key_len == value_len

        scaling = float(self.head_dim) ** -0.5

        # ------------------------------------------------------------------------------------------------ #
        # Compute q, k, v
        # ------------------------------------------------------------------------------------------------ #
        if self._qkv_same_embed_dim:
            # self-attention
            if torch.equal(query, key) and torch.equal(key, value):
                qkv = self.in_proj(query)
                q, k, v = qkv.chunk(3, dim=-1)

            # enc-dec attention
            elif torch.equal(key, value):
                q = F.linear(query, self.in_proj.weight()[:embed_dim, :],
                             self.in_proj.bias()[:embed_dim] if (self.in_proj.bias is not None) else None)
                if key is None:
                    assert value is None
                    k = v = None
                else:
                    kv = F.linear(key, self.in_proj.weight()[embed_dim:, :],
                                  self.in_proj.bias()[embed_dim:] if (self.in_proj.bias is not None) else None)
                    k, v = kv.chunk(2, dim=-1)

            # general attention
            else:
                q = F.linear(query, self.in_proj.weight()[:embed_dim, :],
                             self.in_proj.bias()[:embed_dim] if (self.in_proj.bias is not None) else None)
                k = F.linear(key, self.in_proj.weight()[embed_dim:embed_dim * 2, :],
                             self.in_proj.bias()[embed_dim:embed_dim * 2] if (self.in_proj.bias is not None) else None)
                v = F.linear(value, self.in_proj.weight()[embed_dim * 2:, :],
                             self.in_proj.bias()[embed_dim * 2:] if (self.in_proj.bias is not None) else None)

        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * scaling

        # ------------------------------------------------------------------------------------------------ #
        # Check attention mask
        # ------------------------------------------------------------------------------------------------ #
        if attn_mask is not None:
            if attn_mask.ndim == 2:  # (query_length, key_length)
                attn_mask = attn_mask.unsqueeze(0)  # (1, query_length, key_length)
                if attn_mask.shape != (1, target_len, key_len):
                    raise ValueError("[ERROR:NN] Attention mask 2D size invalid.")
            elif attn_mask.ndim == 3:  # (batch_size * num_heads, query_length, key_length)
                if attn_mask.shape != (batch_size * self.num_heads, target_len, key_len):
                    raise ValueError("[ERROR:NN] Attention mask 3D size invalid.")
            else:
                raise RuntimeError("[ERROR:NN] Attention mask should be either 2D or 3D.")

        # ------------------------------------------------------------------------------------------------ #
        # Add bias of KV
        # ------------------------------------------------------------------------------------------------ #
        if (self.bias_k is not None) and (self.bias_v is not None):
            # treat bias as additional sequence input
            k = torch.cat([k, self.bias_k().view(1, 1, -1).repeat(1, batch_size, 1)])
            v = torch.cat([v, self.bias_v().view(1, 1, -1).repeat(1, batch_size, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))  # extend last key_length += 1
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))  # extend last key_length += 1

        # ------------------------------------------------------------------------------------------------ #
        # Dot product
        # ------------------------------------------------------------------------------------------------ #
        q = q.contiguous().view(target_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.shape[1]  # may be updated because of bias_k & bias_v.
        assert v.shape[1] == src_len

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, src_len)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros(k.shape[0], 1, k.shape[2], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(v.shape[0], 1, v.shape[2], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (batch_size * num_heads, target_len, src_len)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask  # expect to be filled with -inf.

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(batch_size, self.num_heads, target_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.view(batch_size, 1, 1, src_len),
                                                                  float("-inf"))
            attn_output_weights = attn_output_weights.view(batch_size * self.num_heads, target_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # (batch_size * num_heads, target_len, src_len)
        if self.drop_prob > 0:
            # drop attention output probabilities
            attn_output_weights = F.dropout(attn_output_weights, p=self.drop_prob, training=self.training,
                                            inplace=False)

        attn_output = torch.bmm(attn_output_weights, v)  # (batch_size * num_heads, target_len, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, batch_size, embed_dim)

        # ------------------------------------------------------------------------------------------------ #
        # Output projection
        # ------------------------------------------------------------------------------------------------ #
        proj_output = self.out_proj(attn_output)  # (target_len, batch_size, embed_dim)

        if need_weights:
            # averaged weights over heads  (batch_size, target_len, src_len)
            attn_output_weights = attn_output_weights.view(batch_size, self.num_heads, target_len, src_len).mean(dim=1)
            return proj_output, attn_output_weights
        else:
            return proj_output, None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                *, key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attn_core(query, key, value, key_padding_mask, need_weights, attn_mask)

    def extra_repr(self) -> str:
        s = f"{self.embed_dim}, num_heads:{self.num_heads}"
        if self.drop_prob > 0:
            s += f", drop_prob={self.drop_prob}"
        if not self.use_bias:
            s += f", bias=False"
        if self.bias_k is not None:
            s += f", add_bias_kv=True"
        if self.add_zero_attn:
            s += f", add_zero_attn=True"
        if not self._qkv_same_embed_dim:
            s += f", kdim={self.kdim}, vdim={self.vdim}"
        return s
