from typing import Optional
import torch

_LARGE_NEGATIVE_VALUE = -1e4  # consider fp16 range [-65504 ~ 65504]


def attn_mask_reshape(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, n, q_len, k_len = attn.shape

    mask = mask.bool()

    if mask.ndim == 2:
        if mask.shape != (q_len, k_len):
            raise ValueError(f"[ERROR:NN] Attention mask 2D shape invalid. "
                             f"Should be ({q_len}, {k_len}) but got {mask.shape}.")
        mask = mask.view(1, 1, q_len, k_len)  # (1, 1, q_len, k_len)

    elif mask.ndim == 3:
        if mask.shape != (b, q_len, k_len):
            raise ValueError(f"[ERROR:NN] Attention mask 3D shape invalid. "
                             f"Should be ({b}, {q_len}, {k_len}) but got {mask.shape}.")
        mask = mask.view(b, 1, q_len, k_len)  # (b, 1, q_len, k_len)

    elif mask.ndim == 4:
        if (mask.shape != (b, 1, q_len, k_len)) or (mask.shape != (b, n, q_len, k_len)):
            raise ValueError(f"[ERROR:NN] Attention mask 4D shape invalid. "
                             f"Should be ({b}, {n} (or 1), {q_len}, {k_len}) but got {mask.shape}.")
    else:
        raise ValueError(f"[ERROR:NN] Unsupported mask dimension, should be either 2/3/4D, but got {mask.ndim}.")
    return mask


def bmm_4d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, h, a, d = x.shape
    _, _, _, c = y.shape
    assert tuple(y.shape) == (b, h, d, c)

    out = torch.bmm(x.view(b * h, a, d), y.view(b * h, d, c)).view(b, h, a, c)
    return out


def bmm_4d_transpose(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, h, q, d = x.shape
    _, _, k, _ = y.shape
    assert tuple(y.shape) == (b, h, k, d)

    out = torch.bmm(x.view(b * h, q, d), y.view(b * h, k, d).transpose(1, 2)).view(b, h, q, k)
    return out


def safe_softmax(x: torch.Tensor,
                 dim: int = -1,
                 *, mask: Optional[torch.Tensor]=None,
                 eps: float = 1e-8) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/issues/55056
    x = x.float()
    with torch.no_grad():
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_max = torch.masked_fill(x_max, torch.eq(x_max, float("-inf")), 0.0).detach()  # (-inf) - (-inf) = nan

    x = x - x_max
    exp_x = torch.exp(x)
    denominator = torch.sum(exp_x, dim=dim, keepdim=True).clamp_min(eps)  # avoid divide-by-zero
    y = exp_x / denominator

    if mask is not None:
        mask = attn_mask_reshape(y, mask)
        y = torch.masked_fill(y, torch.not_equal(mask, 1), 0)
    return y


def apply_attn_mask(attn: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    attn:   (batch_size, num_heads, query_length, key_length), before softmax normalize
    mask:   (query_length, key_length) | (batch_size, query_length, key_length)     bool
        0 -> should be masked
        1 -> should be kept
    """
    if mask is None:
        return attn

    mask = attn_mask_reshape(attn, mask)

    # we intentionally do not masked_fill with -inf, because there is instability issue.
    # https://github.com/pytorch/pytorch/issues/41508

    # add (HuggingFace, AllenNLP)
    # negative_offset = torch.not_equal(mask, 1) * _LARGE_NEGATIVE_VALUE
    # attn += negative_offset  # inplace OK

    # fill (FairSeq), requires safe_softmax
    attn = torch.masked_fill(attn.float(), torch.not_equal(mask, 1), float("-inf"))

    # Or, NVIDIA style
    # https://github.com/NVIDIA/NeMo/blob/5f7651d0e7/nemo/collections/asr/parts/submodules/multi_head_attention.py
    return attn


def apply_weak_attention_suppression(attn: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """WAS. Assume score is BEFORE softmax and AFTER masking.
    score:  (batch_size, num_heads, target_length, source_length)

    Dynamically remove probabilities under (mu - gamma * std)
    Output is BEFORE softmax. This function just mask out small probabilities.
    """

    # compute which values
    with torch.no_grad():
        # score = torch.softmax(attn, dim=-1, dtype=torch.float32)
        score = safe_softmax(attn, dim=-1)

        # we need mean and std over non-zero attention scores (unmasked ones)
        nonzero_count = torch.sum(torch.greater(score, 1e-6), dim=-1, keepdim=True).clamp_min_(1)  # at least 1
        score_sum = torch.sum(score, dim=-1, keepdim=True)
        score_sq_sum = torch.sum(score * score, dim=-1, keepdim=True)
        score_mean = score_sum.div_(nonzero_count)
        score_sq_mean = score_sq_sum.div_(nonzero_count)
        score_std = torch.sqrt(score_sq_mean - (score_mean * score_mean) + 1e-8)

        threshold = score_mean - gamma * score_std
        mask = torch.greater_equal(score, threshold)

    # add (HuggingFace, AllenNLP)
    # negative_offset = torch.not_equal(mask, 1) * _LARGE_NEGATIVE_VALUE
    # attn += negative_offset  # inplace OK

    # fill (FairSeq), requires safe_softmax
    attn = torch.masked_fill(attn.float(), torch.not_equal(mask, 1), float("-inf"))
    return attn
