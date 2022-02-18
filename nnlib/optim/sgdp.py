from typing import Dict, Any
import math
import torch
import torch.nn.functional as F

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappySGDP")
class HappySGDP(BaseOptimizer):
    def __init__(self, params,
                 lr: float = 0.1,
                 momentum: float = 0.0, dampening: float = 0.0,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 delta: float = 0.1,
                 wd_ratio: float = 0.1,
                 nesterov: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid weight_decay value: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid epsilon value: {eps}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("[ERROR:OPT] Nesterov momentum requires a momentum and zero dampening.")
        if delta < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid delta value: {delta}")
        if wd_ratio < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid WD ratio: {wd_ratio}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, eps=eps,
                        weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(HappySGDP, self).__init__(params, defaults)

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
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            delta = group["delta"]
            wd_ratio = group["wd_ratio"]
            nesterov = group["nesterov"]

            # ---------------------------------------------------------------- #
            # dW = -lr * grad

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # -------------------------------- #
                # Gradient momentum
                # -------------------------------- #
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        d_p = grad.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                    # -------------------------------- #
                    # Weight decay
                    # -------------------------------- #
                    if weight_decay != 0:
                        if len(p.shape) > 1:  # 2D
                            d_p, wd_ratio = self._projection(p, grad, d_p, delta, wd_ratio, eps)
                        else:
                            wd_ratio = 1
                        p.mul_(1 - lr * weight_decay * wd_ratio / (1 - momentum))
                else:
                    d_p = grad

                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                # -------------------------------- #
                # Update
                # -------------------------------- #
                p.add_(d_p, alpha=-lr)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappySGDP":
        return cls(
            params=params,
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.0),
            dampening=config.get("dampening", 0.0),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            delta=config.get("delta", 0.1),
            wd_ratio=config.get("wd_ratio", 0.1),
            nesterov=config.get("nesterov", False)
        )
