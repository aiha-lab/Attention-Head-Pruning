from typing import Tuple
import torch

from .print_utils import print_log


def count_params(parameters) -> Tuple[int, int]:
    """Count (#tensors, #elements)"""
    count = 0
    count_elements = 0
    for p in parameters:
        p: torch.Tensor
        if not p.requires_grad:
            continue
        count += 1
        count_elements += p.numel()
    return count, count_elements


def compute_param_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.requires_grad]
    device = parameters[0].device
    if len(parameters) == 0:
        return torch.as_tensor(0., device=device)
    with torch.no_grad():
        total_norm = torch.norm(torch.stack([torch.norm(p, norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def print_params(model: torch.nn.Module) -> None:
    s = "-" * 72 + "\n"
    s += "Parameters:\n"
    for param_name, param in model.named_parameters():
        s += f"... {param_name:<60}\t{tuple(param.shape)}\n"
    s += "-" * 72
    print_log(s)
