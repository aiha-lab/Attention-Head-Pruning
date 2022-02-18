from typing import Any, Optional
import torch

from nnlib.utils.dist_utils import all_gather_tensor, all_reduce_tensor, get_world_size

from torch.autograd import Function


class PACTFunc(Function):
    # Some useful extensions:
    # https://github.com/cornell-zhang/dnn-gating/blob/master/utils/pg_utils.py

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inp, alpha)
        res = torch.clamp(inp, 0, alpha.item())
        return res

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        inp, alpha = ctx.saved_tensors

        grad_input = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[inp < 0] = 0
            grad_input[inp > alpha] = 0
        if ctx.needs_input_grad[1]:
            grad_alpha = grad_output.clone()
            grad_alpha[inp <= alpha] = 0
            grad_alpha = torch.sum(grad_alpha, dim=0, keepdim=True)  # to preserve [1],
        return grad_input, grad_alpha


pact = PACTFunc.apply


class GradientScaleFunc(Function):
    # forward as-is, backward gradient scaling

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return inp

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        scale = ctx.saved_tensors[0]
        grad_output.mul_(scale)
        return grad_output, None


gradient_scale = GradientScaleFunc.apply


class MaskedBatchNormFunc(Function):
    """Should be only called when bn_training=True."""

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor,
                weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                mask: torch.Tensor,
                running_mean: Optional[torch.Tensor], running_var: Optional[torch.Tensor],
                momentum: float, eps: float) -> torch.Tensor:
        # c = inp.shape[1]

        # mask should be in broadcast-able shape as input
        assert mask is not None
        assert tuple(mask.shape) == (inp.shape[0],) + (1,) + inp.shape[2:]
        inp = inp * mask
        count = mask.sum()

        if count == 1:
            raise ValueError

        sum_dim = [0] + list(range(2, inp.ndim))
        mean = torch.sum(inp, dim=sum_dim) / count
        sq_mean = torch.sum(inp * inp, dim=sum_dim) / count
        var = torch.clamp_min(sq_mean - (mean * mean), 1e-6)
        inv_std = torch.rsqrt(var + eps)

        if (running_mean is not None) and (running_var is not None):
            running_mean.data.add_(mean.detach().data - running_mean.data, alpha=momentum)
            running_var.data.add_(var.detach().data - running_var.data, alpha=momentum)

        ctx.save_for_backward(inp, weight, mean, inv_std, mask, count.to(torch.int32))
        output = torch.batch_norm_elemt(inp, weight, bias, mean, inv_std, eps)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        inp, weight, mean, inv_std, mask, count = ctx.saved_tensors

        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            inp,
            mean,
            inv_std,
            weight,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2]
        )

        grad_input = None
        if ctx.needs_input_grad[0]:
            mean_dy = sum_dy / count
            mean_dy_xmu = sum_dy_xmu / count

            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                inp,
                mean,
                inv_std,
                weight,
                mean_dy,
                mean_dy_xmu,
            )
            grad_input *= mask

        if (weight is None) or (not ctx.needs_input_grad[1]):
            grad_weight = None
        if (weight is None) or (not ctx.needs_input_grad[2]):  # isn't this should be bias?
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


masked_batch_norm_func = MaskedBatchNormFunc.apply


class SyncBatchNormFunc(Function):
    """Should be only called when bn_training=True."""

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor,
                weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                running_mean: Optional[torch.Tensor], running_var: Optional[torch.Tensor],
                momentum: float, eps: float) -> torch.Tensor:
        c = inp.shape[1]
        size = inp.numel() // c  # elements to be summed

        if (size == 1) and (get_world_size() < 2):
            raise ValueError

        mean, inv_std = torch.batch_norm_stats(inp, eps)
        count = torch.full((1,), size, dtype=mean.dtype, device=mean.device)

        combined = torch.cat([mean, inv_std, count], dim=0)  # (2C + 1,)
        combined = all_gather_tensor(combined)
        combined = torch.stack(combined, dim=0)  # (world_size, 2C + 1)
        mean_all, inv_std_all, count_all = torch.split(combined, c, 1)  # (world, C), (world, C), (world, 1)

        mean, inv_std = torch.batch_norm_gather_stats_with_counts(
            inp,
            mean_all,
            inv_std_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )
        ctx.save_for_backward(inp, weight, mean, inv_std, count_all.to(torch.int32))
        output = torch.batch_norm_elemt(inp, weight, bias, mean, inv_std, eps)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        inp, weight, mean, inv_std, count = ctx.saved_tensors
        c = inp.shape[1]

        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            inp,
            mean,
            inv_std,
            weight,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2]
        )

        grad_input = None
        if ctx.needs_input_grad[0]:
            # synchronize stats
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)  # (2C,)
            combined = all_reduce_tensor(combined, "sum", detach=False)
            sum_dy, sum_dy_xmu = torch.split(combined, c)

            divisor = count.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor

            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                inp,
                mean,
                inv_std,
                weight,
                mean_dy,
                mean_dy_xmu,
            )

        if (weight is None) or (not ctx.needs_input_grad[1]):
            grad_weight = None
        if (weight is None) or (not ctx.needs_input_grad[2]):  # isn't this should be bias?
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


sync_batch_norm_func = SyncBatchNormFunc.apply


class MaskedSyncBatchNormFunc(Function):
    """Should be only called when bn_training=True."""

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor,
                weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                mask: torch.Tensor,
                running_mean: Optional[torch.Tensor], running_var: Optional[torch.Tensor],
                momentum: float, eps: float) -> torch.Tensor:
        c = inp.shape[1]

        # mask should be in broadcast-able shape as input
        assert mask is not None
        assert tuple(mask.shape) == (inp.shape[0],) + (1,) + inp.shape[2:]
        inp = inp * mask
        count = mask.sum()  # assume mask shape same as input, except channel dimension.

        if (count == 1) and (get_world_size() < 2):
            raise ValueError

        sum_dim = [0] + list(range(2, inp.ndim))
        mean = torch.sum(inp, dim=sum_dim) / count
        sq_mean = torch.sum(inp * inp, dim=sum_dim) / count
        var = torch.clamp_min(sq_mean - (mean * mean), 1e-6)
        inv_std = torch.rsqrt(var + eps)

        combined = torch.cat([mean, inv_std, count.view(1)], dim=0)  # (2C + 1,)
        combined = all_gather_tensor(combined)
        combined = torch.stack(combined, dim=0)  # (world_size, 2C + 1)
        mean_all, inv_std_all, count_all = torch.split(combined, c, 1)  # (world, C), (world, C), (world, 1)

        mean, inv_std = torch.batch_norm_gather_stats_with_counts(
            inp,
            mean_all,
            inv_std_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )
        ctx.save_for_backward(inp, weight, mean, inv_std, mask, count_all.to(torch.int32))
        output = torch.batch_norm_elemt(inp, weight, bias, mean, inv_std, eps)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        inp, weight, mean, inv_std, mask, count = ctx.saved_tensors
        c = inp.shape[1]

        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            inp,
            mean,
            inv_std,
            weight,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2]
        )

        grad_input = None
        if ctx.needs_input_grad[0]:
            # synchronize stats
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)  # (2C,)
            combined = all_reduce_tensor(combined, "sum", detach=False)
            sum_dy, sum_dy_xmu = torch.split(combined, c)  # (C,), (C,)

            divisor = count.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor

            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                inp,
                mean,
                inv_std,
                weight,
                mean_dy,
                mean_dy_xmu,
            )
            grad_input *= mask

        if (weight is None) or (not ctx.needs_input_grad[1]):
            grad_weight = None
        if (weight is None) or (not ctx.needs_input_grad[2]):  # isn't this should be bias?
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


masked_sync_batch_norm_func = MaskedSyncBatchNormFunc.apply
