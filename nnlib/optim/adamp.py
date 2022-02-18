from typing import Tuple, Dict, Any
import math
import torch
import torch.nn.functional as F

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappyAdamP")
class HappyAdamP(BaseOptimizer):

    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 delta: float = 0.1,
                 wd_ratio: float = 0.1,
                 nesterov: bool = False) -> None:
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
        if delta < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid delta value: {delta}")
        if wd_ratio < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid WD ratio: {wd_ratio}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio,
                        nesterov=nesterov)
        super(HappyAdamP, self).__init__(params, defaults)

    @staticmethod
    def _channel_view(x):
        return x.view(x.shape(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    @staticmethod
    def _cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)
        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio
                return perturb, wd

        return perturb, wd

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
            delta = group["delta"]
            wd_ratio = group["wd_ratio"]
            nesterov = group["nesterov"]

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

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denominator
                else:
                    perturb = exp_avg / denominator

                # -------------------------------- #
                # Weight decay
                # -------------------------------- #
                if weight_decay != 0:
                    if len(p.shape) > 1:  # 2D
                        perturb, wd_ratio = self._projection(p, grad, perturb, delta, wd_ratio, eps)
                    else:
                        wd_ratio = 1
                    p.mul_(1 - lr * weight_decay * wd_ratio)

                p.add_(perturb, alpha=-step_size)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappyAdamP":
        return cls(
            params=params,
            lr=config.get("lr", 0.001),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            delta=config.get("delta", 0.1),
            wd_ratio=config.get("wd_ratio", 0.1),
            nesterov=config.get("nesterov", False)
        )
