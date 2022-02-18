import torch
import math


class DummyGradScaler(object):
    """Do nothing, just provides same interface as amp.GradScaler."""

    @staticmethod
    def scale(outputs):
        return outputs

    @staticmethod
    def unscale_(optimizer):
        return

    @staticmethod
    def step(optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    @staticmethod
    def update(new_scale=None):
        return

    @staticmethod
    def get_scale() -> float:
        return 1.0

    @staticmethod
    def is_enabled() -> bool:
        return False

    @staticmethod
    def state_dict():
        return list()

    @staticmethod
    def load_state_dict(state_dict):
        return


def compute_grad_norm(parameters, clip_value: float = 0.0, norm_type: float = 2.0) -> torch.Tensor:
    """Compute gradient norm and clip grad norm if clip_value > 0"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    if clip_value > 0.0:
        clip_coefficient = clip_value / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coefficient.to(p.grad.device))
    return total_norm
