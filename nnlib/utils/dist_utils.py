from typing import Dict, Optional, Any, List
from numbers import Number
import datetime
import io
import os
import torch
import torch.distributed as dist


def init_distributed(cuda: bool = True, *,
                     init_method: str = "env://",
                     timeout=datetime.timedelta(minutes=3)) -> bool:
    # os.environ["NCCL_BLOCKING_WAIT"] = str(1)  # FOR DEBUG
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_distributed = (world_size > 1)
    if use_distributed:
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("[ERROR:DIST] WORLD_SIZE is set but LOCAL_RANK is not. Use `launch` to instantiate DDP.")
        backend = "nccl" if cuda else "gloo"
        local_rank = int(os.environ.get("LOCAL_RANK"))
        assert local_rank is not None
        dist.init_process_group(backend=backend,
                                init_method=init_method,  # default: "env://"
                                world_size=world_size,
                                rank=local_rank,
                                timeout=timeout  # default: timedelta(minutes=30) for pytorch.
                                )
        assert dist.is_initialized()
    return use_distributed


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_master() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_scalar(value: Number, op: str = "sum") -> Number:
    """All-reduce single scalar value. NOT torch tensor."""
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py
    if dist.is_available() and dist.is_initialized():
        op = op.lower()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        elif op == "min":
            dist_op = dist.ReduceOp.MIN
        elif op == "max":
            dist_op = dist.ReduceOp.MAX
        elif op == "product":
            dist_op = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError(f"[ERROR:DIST] Invalid distributed op type: {op}")

        backend = dist.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError(f"[ERROR:DIST] Unsupported distributed backend: {backend}")

        tensor = torch.tensor(value, device=device, requires_grad=False)
        dist.all_reduce(tensor, op=dist_op)
        if op == "mean":
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret


def all_reduce_tensor(tensor: torch.Tensor, op="sum", detach: bool = True) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        ret = tensor.clone()
        if detach:
            ret = ret.detach()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        else:
            raise RuntimeError(f"[ERROR:DIST] Invalid distributed op: {op}")

        dist.all_reduce(ret, op=dist_op)
        if op == "mean":
            ret /= get_world_size()
    else:
        ret = tensor
    return ret


def all_reduce_dict(result: Dict[str, Any], op="sum") -> Dict[str, Any]:
    new_result = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            new_result[k] = all_reduce_tensor(v, op)
        elif isinstance(v, Number):
            new_result[k] = all_reduce_scalar(v, op)
        else:
            raise RuntimeError(f"[ERROR:DIST] All-reduce-dict should only have either tensor or scalar.")
    return new_result


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    if dist.is_available() and dist.is_initialized():
        world_size = get_world_size()
        local_rank = get_rank()
        output = [
            tensor if (i == local_rank) else torch.empty_like(tensor) for i in range(world_size)
        ]
        dist.all_gather(output, tensor, async_op=False)
        return output
    else:
        return [tensor]


def _broadcast_object(obj: Any, src_rank, device) -> Any:
    # see FairSeq/distributed/utils
    if src_rank == get_rank():
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(device)
        length = torch.LongTensor([len(buffer)]).to(device)
        dist.broadcast(length, src=src_rank)
        dist.broadcast(buffer, src=src_rank)
    else:
        length = torch.LongTensor([0]).to(device)
        dist.broadcast(length, src=src_rank)
        buffer = torch.ByteTensor(int(length.item())).to(device)
        dist.broadcast(buffer, src=src_rank)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location="cpu")
    return obj


def broadcast_objects(obj_list: List[Any], src_rank: int = 0) -> List[Any]:
    # list should have same length
    # dist.broadcast_object_list(obj_list, src=src_rank)  # somehow not working
    backend = torch.distributed.get_backend()
    if backend == torch.distributed.Backend.NCCL:
        device = torch.device("cuda")
    elif backend == torch.distributed.Backend.GLOO:
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"[ERROR:DIST] Unsupported distributed backend: {backend}")

    out = []
    for obj in obj_list:
        out.append(_broadcast_object(obj, src_rank, device=device))
    return out


def broadcast_tensor(tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    # tensor should have same number of elements and dtype through GPUs
    dist.broadcast(tensor, src=src_rank)
    return tensor


def broadcast_tensor_any(tensor: Optional[torch.Tensor], src_rank: int = 0) -> torch.Tensor:
    # see FairSeq/distributed/utils
    # broadcast, not restricted to tensor shape and dtype match.
    device = torch.device("cuda")

    if src_rank == get_rank():
        if tensor is None:
            raise RuntimeError(f"[ERROR:DIST] Broadcast tensor in src_rank, but got None.")
        metadata = {"shape": tensor.shape, "dtype": tensor.dtype}
        metadata = _broadcast_object(metadata, src_rank, device)
    else:
        metadata = _broadcast_object(None, src_rank, device)

    if src_rank == get_rank():
        dist.broadcast(tensor, src=src_rank)
    else:
        tensor = torch.zeros(*metadata["shape"], dtype=metadata["dtype"], device=device)
        dist.broadcast(tensor, src=src_rank)
    return tensor
