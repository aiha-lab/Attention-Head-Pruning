from typing import Optional
import math
import torch

from nnlib.nn.modules import BaseModule, Embedding


class _SinusoidalPositionalEncodingBase(BaseModule):

    def __init__(self,
                 embed_dim: int,
                 clamp_length: Optional[int] = None,
                 padding_idx: Optional[int] = 0, *,
                 inverse: bool = False):
        super(_SinusoidalPositionalEncodingBase, self).__init__()

        if embed_dim % 2 != 0:
            raise ValueError(f"[ERROR:NN] Currently SinusoidalPE only supports even embed_dim.")
        self.embed_dim = embed_dim
        self.clamp_length = clamp_length
        self.padding_idx = padding_idx
        self.inverse = inverse

        # inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        inv_freq = torch.exp(
            torch.arange(0.0, embed_dim, 2.0, dtype=torch.float32).mul_(-math.log(10000.0) / embed_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (embed_dim // 2,)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = f"{self.embed_dim}"
        if self.clamp_length is not None:
            s += f", clamp_length={self.clamp_length}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.inverse:
            s += f", inverse=True"
        return s


class SinusoidalPositionalEncoding(_SinusoidalPositionalEncodingBase):

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices:        (batch_size, seq_len, ...)     same interface as word embedding.
        Return:         (batch_size, seq_len, embed_dim)
                            [0, 1, 2, 3, ... s-1] if not inverse
                            [s-1, s-2, .... 1, 0] if inverse
        """
        # convert indices to position
        batch_size, seq_len = indices.shape[:2]
        with torch.no_grad():
            device = self.inv_freq.device
            emb = torch.zeros(seq_len, self.embed_dim, dtype=torch.float32, device=device)
            pos = torch.arange(0, seq_len, dtype=torch.float32, device=device)
            if self.clamp_length is not None:
                pos = pos.clamp_max_(self.clamp_length - 1)
            dot = torch.outer(pos, self.inv_freq)
            emb[:, 0::2] = torch.sin(dot)
            emb[:, 1::2] = torch.cos(dot)

            if self.inverse:
                # [0, 1, 2, 3, ... seq_len - 1] -> [seq_len - 1, ... 1, 0]
                emb = torch.flipud(emb).contiguous()

            emb = emb.unsqueeze(0).expand(batch_size, seq_len, self.embed_dim)
            if self.padding_idx is not None:
                mask = torch.not_equal(indices, self.padding_idx).unsqueeze(2)  # (b, s, 1)
                emb = emb * mask

        return emb


class BidirectionalSinusoidalPositionalEncoding(_SinusoidalPositionalEncodingBase):

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices:        (batch_size, seq_len, ...)     same interface as word embedding.
        Return:         (batch_size, seq_len * 2, embed_dim)
                            [-s, -s+1, -s+2, ... 0, 1, 2, 3, ... s-1] if not inverse
                            [s-1, s-2, ... 2, 1, 0, -1, -2, .... -s] if inverse (shouldn't required for Bidirectional)
        """
        # convert indices to position
        batch_size, seq_len = indices.shape[:2]
        with torch.no_grad():
            device = self.inv_freq.device
            emb = torch.zeros(seq_len * 2, self.embed_dim, dtype=torch.float32, device=device)
            pos = torch.arange(-seq_len, seq_len, dtype=torch.float32, device=device)
            if self.clamp_length is not None:
                pos = pos.clamp_(-self.clamp_length, self.clamp_length - 1)

            dot = torch.outer(pos, self.inv_freq)
            emb[:, 0::2] = torch.sin(dot)
            emb[:, 1::2] = torch.cos(dot)

            if self.inverse:
                emb = torch.flipud(emb).contiguous()

            emb = emb.unsqueeze(0).expand(batch_size, seq_len * 2, self.embed_dim)
            if self.padding_idx is not None:
                mask = torch.not_equal(indices, self.padding_idx).unsqueeze(2)  # (b, s, 1)
                emb[: -seq_len:] = emb[:, -seq_len] * mask

        return emb


class LearnedPositionalEncoding(Embedding):

    def __init__(self,
                 embed_dim: int,
                 max_seq_length: int,
                 padding_idx: int = 0, *,
                 word_drop_prob: float = 0.0):
        if padding_idx is None:
            raise ValueError(f"[ERROR:NN] LearnedPE should give integer padding_idx, got None.")

        super(LearnedPositionalEncoding, self).__init__(max_seq_length, embed_dim,
                                                        padding_idx=padding_idx, word_drop_prob=word_drop_prob)
        self.max_len = max_seq_length - (padding_idx + 1)  # ignore below padding_idx.

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices:      (batch_size, seq_len, ...)
        Return:
            pos_emb:    (batch_size, seq_len, embed_dim)
        """
        # convert indices to position
        batch_size, seq_len = indices.shape[:2]
        if seq_len > self.max_len:
            raise RuntimeError(f"[ERROR:NN] LearnedPE length overflow: maximum {self.max_len}, got {seq_len}.")

        # see FairSeq/utils/make_positions
        mask = torch.not_equal(indices, self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) * mask).long() + self.padding_idx

        return super(LearnedPositionalEncoding, self).forward(positions)
