from typing import Dict, Any
import torch

from nnlib.optim import register_optimizer
from nnlib.optim.optimizer import BaseOptimizer


@register_optimizer("HappyRMSProp")
class HappyRMSProp(BaseOptimizer):

    def __init__(self, params,
                 lr: float = 1e-2,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum: float = 0,
                 centered: bool = False,
                 decoupled: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid learning rate: {lr}")
        if not 0.0 <= alpha:
            raise ValueError(f"[ERROR:OPT] Invalid alpha value: {alpha}")
        if eps <= 0.0:
            raise ValueError(f"[ERROR:OPT] Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"[ERROR:OPT] Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"[ERROR:OPT] Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        weight_decay=weight_decay, centered=centered, decoupled=decoupled)
        super(HappyRMSProp, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            momentum = group["momentum"]
            centered = group["centered"]
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
                    state["square_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if centered:
                        state["grad_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state["square_avg"]

                state["step"] += 1

                # -------------------------------- #
                # Update
                # -------------------------------- #
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
                else:
                    avg = square_avg.sqrt().add_(eps)

                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-lr)
                else:
                    p.addcdiv_(grad, avg, value=-lr)

        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any], params) -> "HappyRMSProp":
        return cls(
            params=params,
            lr=config.get("lr", 0.001),
            alpha=config.get("alpha", 0.99),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            momentum=config.get("momentum", 0.0),
            centered=config.get("centered", False),
            decoupled=config.get("decoupled", False)
        )
