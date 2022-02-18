from typing import Tuple, Dict, Any
import torch
import math

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappyLamb")
class HappyLamb(BaseOptimizer):
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0,
                 adam: bool = False,
                 debiasing: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"[ERROR:OPT] Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"[ERROR:OPT] Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, adam=adam, debiasing=debiasing)
        super(HappyLamb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            adam = group["adam"]
            eps = group["eps"]
            debiasing = group["debiasing"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # -------------------------------- #
                # Gradient momentum
                # -------------------------------- #
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # -------------------------------- #
                # Update
                # -------------------------------- #
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if debiasing:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                else:
                    denominator = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr

                adam_step = exp_avg / denominator
                if weight_decay != 0:
                    adam_step.add_(p, alpha=weight_decay)

                adam_norm = adam_step.norm(p=2)
                weight_norm = p.norm(p=2).clamp_(0, 10)
                if (weight_norm == 0.0) or (adam_norm == 0.0):
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / (adam_norm + eps)

                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if adam:
                    trust_ratio = 1

                p.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappyLamb":
        return cls(
            params=params,
            lr=config.get("lr", 0.001),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            adam=config.get("adam", False),
            debiasing=config.get("debiasing", False)
        )


# @torch.jit.script
# def lamb_kernel(param, grad, exp_avg, exp_avg_sq,
#                 beta1: float, beta2: float, step_size: float, eps: float, weight_decay: float):
#     exp_avg = exp_avg * beta1 + (1 - beta1) * grad
#     exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * (grad * grad)
#
#     adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
#     adam_step = adam_step + weight_decay * param
#
#     weight_norm = param.norm(p=2).clamp(0, 10)
#     adam_norm = adam_step.norm(p=2)
#
#     trust_ratio = weight_norm / (adam_norm + eps)
#     trust_ratio = (weight_norm == 0.0) * 1.0 + (weight_norm != 0.0) * trust_ratio
#     trust_ratio = (adam_norm == 0.0) * 1.0 + (adam_norm != 0.0) * trust_ratio
#     trust_ratio = trust_ratio.float()
#
#     param = param - step_size * trust_ratio * adam_step
#     return param, exp_avg, exp_avg_sq
