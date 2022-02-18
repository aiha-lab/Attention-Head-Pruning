from typing import Tuple, Dict, Any
import math
import torch

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappyRAdam")
class HappyRAdam(BaseOptimizer):
    """https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
    Following `PlainRAdam` implementation, which is also used for https://github.com/LiyuanLucasLiu/Transformer-Clinic.
    """

    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 decoupled: bool = True,
                 degenerated_to_sgd: bool = True) -> None:
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
                        weight_decay=weight_decay, decoupled=decoupled,
                        degenerated_to_sgd=degenerated_to_sgd)
        super(HappyRAdam, self).__init__(params, defaults)

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
            eps = group["eps"]
            decoupled = group["decoupled"]
            degenerated_to_sgd = group["degenerated_to_sgd"]

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

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # -------------------------------- #
                # Update
                # -------------------------------- #
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # rectified update
                beta2_t = 1 - bias_correction2
                n_sma_max = 2 / (1 - beta2) - 1
                n_sma = n_sma_max - 2 * state["step"] * beta2_t / bias_correction2

                if n_sma >= 5:
                    step_size = math.sqrt(bias_correction2 * (n_sma - 4) / (n_sma_max - 4) *
                                          (n_sma - 2) / (n_sma_max - 2) *
                                          n_sma_max / n_sma) / bias_correction1
                    denominator = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denominator, value=-step_size * lr)
                elif degenerated_to_sgd:
                    step_size = 1.0 / bias_correction1
                    p.add_(exp_avg, alpha=-step_size * lr)
                else:
                    # step_size = -1
                    pass  # nothing to update

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappyRAdam":
        return cls(
            params=params,
            lr=config.get("lr", 0.001),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            decoupled=config.get("decoupled", True),
            degenerated_to_sgd=config.get("degenerated_to_sgd", True)
        )
