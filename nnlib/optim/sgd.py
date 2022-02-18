from typing import Dict, Any
import torch

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappySGD")
class HappySGD(BaseOptimizer):
    def __init__(self, params,
                 lr: float = 0.1,
                 momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0,
                 nesterov: bool = False, decoupled: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(f"[ERROR:OPT] Nesterov momentum requires a momentum and zero dampening.")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, decoupled=decoupled)
        super(HappySGD, self).__init__(params, defaults)

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
            nesterov = group["nesterov"]
            decoupled = group["decoupled"]

            # ---------------------------------------------------------------- #
            # dW = -lr * grad

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # -------------------------------- #
                # Weight decay
                # -------------------------------- #
                if weight_decay != 0:
                    if decoupled:
                        p.mul_(1 - lr * weight_decay)
                    else:
                        d_p = d_p.add(p, alpha=weight_decay)

                # -------------------------------- #
                # Gradient momentum
                # -------------------------------- #
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # -------------------------------- #
                # Update
                # -------------------------------- #
                p.add_(d_p, alpha=-lr)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappySGD":
        return cls(
            params=params,
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.0),
            dampening=config.get("dampening", 0.0),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
            decoupled=config.get("decoupled", False)
        )
