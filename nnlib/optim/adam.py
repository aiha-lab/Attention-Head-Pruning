from typing import Tuple, Dict, Any
import math
import torch

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappyAdam")
class HappyAdam(BaseOptimizer):

    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 decoupled: bool = False) -> None:
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
                        weight_decay=weight_decay, amsgrad=amsgrad, decoupled=decoupled)
        super(HappyAdam, self).__init__(params, defaults)

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
            amsgrad = group["amsgrad"]
            eps = group["eps"]
            decoupled = group["decoupled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # -------------------------------- #
                # Weight decay
                # -------------------------------- #
                if weight_decay != 0:
                    if decoupled:
                        p.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p, alpha=weight_decay)

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
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # -------------------------------- #
                # Update
                # -------------------------------- #
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denominator = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappyAdam":
        return cls(
            params=params,
            lr=config.get("lr", 0.001),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            amsgrad=config.get("amsgrad", False),
            decoupled=config.get("decoupled", False)
        )
