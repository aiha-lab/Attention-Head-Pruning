from typing import Optional
import math
import torch
import torch.cuda.amp as amp

from nnlib.nn.modules import BaseModule
from nnlib.nn.transformer.utils import (apply_attn_mask, apply_weak_attention_suppression,
                                        bmm_4d_transpose, safe_softmax)


class ScaledDotProduct(BaseModule):
    """Compute attention score (matrix). If normalize=True, apply Softmax to output."""

    def __init__(self,
                 scaling_factor: int, *,
                 normalize: bool = True,
                 was: bool = False,
                 was_gamma: float = 0.5):
        super(ScaledDotProduct, self).__init__()
        self.scaling_factor = scaling_factor
        self.scaling = float(1 / math.sqrt(scaling_factor))
        self.normalize = normalize

        self.was = was
        self.was_gamma = was_gamma

    def reset_scaling(self, new_scaling_factor: int):
        self.scaling_factor = new_scaling_factor
        self.scaling = float(1 / math.sqrt(new_scaling_factor))

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ W = softmax(QK^T / sqrt(k))
        Q:          (batch_size, num_heads, query_length, head_dim)
        K:          (batch_size, num_heads, key_length, head_dim)
        mask:       (batch_size, query_length, key_length)     bool

        Return
            attn:   (batch_size, num_heads, query_length, key_length)
        """
        q_len, dq = query.shape[-2:]
        k_len, dk = key.shape[-2:]
        if (dq != dk) or (query.ndim != key.ndim):
            raise ValueError(f"[ERROR:NN] Query {query.shape}, Key {key.shape} shape mismatch.")

        with amp.autocast(enabled=False):
            query = query.float()
            key = key.float()

            if query.ndim == 3:
                attn = torch.bmm(query, key.transpose(1, 2))  # (b, q_len, k_len)
            elif query.ndim == 4:
                # attn = torch.einsum("bhqd,bhkd->bhqk", query, key)  # (b, h, q_len, k_len)
                attn = bmm_4d_transpose(query, key)  # (b, h, q_len, k_len)
            else:
                raise ValueError(f"[ERROR:NN] Q/K ndim should be 3/4D, got Q: {query.shape}, K: {key.shape}.")

            attn *= self.scaling  # inplace OK
            attn = apply_attn_mask(attn, mask)
            if self.was:
                attn = apply_weak_attention_suppression(attn, gamma=self.was_gamma)
            if self.normalize:
                # attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
                attn = safe_softmax(attn, dim=-1, mask=mask)
        return attn

    def extra_repr(self) -> str:
        s = f"scaling_factor={self.scaling_factor}"
        if not self.normalize:
            s += ", normalize=False"
        if self.was:
            s += f", was=True, was_gamma={self.was_gamma}"
        return s
