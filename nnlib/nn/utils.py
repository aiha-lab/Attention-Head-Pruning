from typing import List, Optional
import torch


def pad_sequence(sequences: List[torch.Tensor],
                 output_batch_first: bool = True,
                 padding_value=0,
                 pad_to_multiple: int = 1) -> torch.Tensor:
    """Unlike torch.nn.utils.pad_sequence, input is always B x (T, ...). """
    batch_size = len(sequences)
    max_len = max([s.shape[0] for s in sequences])
    trailing_dims = sequences[0].shape[1:]

    if (pad_to_multiple > 1) and (max_len % pad_to_multiple != 0):
        max_len += (pad_to_multiple - max_len % pad_to_multiple)
    out_dims = (batch_size, max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor

    if output_batch_first:
        return out_tensor  # (B, T, ...)
    else:
        return out_tensor.transpose(0, 1).contiguous()  # (T, B, ...)


def make_mask_by_length(lengths: torch.Tensor,
                        max_length: Optional[int] = None,
                        right_align: bool = False) -> torch.Tensor:
    """Make boolean T/F mask
    length:     (batch_size,)               long
    mask:       (batch_size, max_length)    bool, True if valid, False if invalid
    """
    if max_length is None:
        max_length = lengths.max().item()

    if lengths.max() > max_length:
        raise ValueError(f"[ERROR:MASK] Maximum length overflow. Got {lengths.max()} but max_length: {max_length}.")

    batch_size = lengths.shape[0]

    with torch.no_grad():
        seq_range = torch.arange(0, max_length, dtype=torch.long, device=lengths.device)  # (s,)
        seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length)  # (b, s)
        seq_length = lengths.unsqueeze(1).expand(batch_size, max_length)  # (b, s)
        mask = torch.less(seq_range, seq_length)
        if right_align:
            mask = torch.fliplr(mask)
    return mask


def make_self_attn_mask_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Make self-attentive mask (outer-product of self)
    mask:           (batch_size, max_length)                bool
    expand_mask:    (batch_size, max_length, max_length)    bool, to self-attentive
    """
    if mask.ndim != 2:
        raise ValueError(f"[ERROR:MASK] mask_expand_self is for 2D tensor (B, T), got {mask.shape}.")

    b, s = mask.shape

    with torch.no_grad():
        mask_f = mask.float()  # bool type is not compatible with matmul
        new_mask = torch.matmul(mask_f.view(b, s, 1), mask_f.view(b, 1, s)).bool()  # (b, s, s)
    return new_mask


def make_cross_attn_mask_from_mask(mask_self: torch.Tensor, mask_cross: torch.Tensor) -> torch.Tensor:
    """Make cross-attentive mask
    mask_self:      (batch_size, max_self_length)   bool
    mask_cross:     (batch_size, max_cross_length)  bool
    expand_mask:    (batch_size, max_self_length, max_cross_length) bool
    """
    if mask_self.ndim != 2:
        raise ValueError(f"[ERROR:MASK] mask_expand_cross is for 2D tensor (B, T), got {mask_self.shape} for self.")
    if mask_cross.ndim != 2:
        raise ValueError(f"[ERROR:MASK] mask_expand_cross is for 2D tensor (B, T), got {mask_cross.shape} for cross.")

    b, s_a = mask_self.shape
    _, s_c = mask_cross.shape
    assert mask_cross.shape[0] == b

    with torch.no_grad():
        mask_s = mask_self.float()
        mask_c = mask_cross.float()
        new_mask = torch.matmul(mask_s.view(b, s_a, 1), mask_c.view(b, 1, s_c)).bool()  # (b, s_a, s_c)
    return new_mask
