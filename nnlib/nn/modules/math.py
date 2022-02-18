from typing import List
import torch

from nnlib.nn.modules.module import BaseModule


class Add(BaseModule):

    def __init__(self):
        super(Add, self).__init__()
        self.num_inputs = 2

    def forward(self, *inputs) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("[ERROR:NN] Add only got 1 input, needs at least 2 inputs.")
        self.num_inputs = len(inputs)
        return sum(inputs)


class Mul(BaseModule):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * x2


class Sub(BaseModule):

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 - x2


class _ScalarMath(BaseModule):

    def __init__(self, val: float):
        super(_ScalarMath, self).__init__()
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"{self.val}"


class ScalarAdd(_ScalarMath):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, self.val)


class ScalarMul(_ScalarMath):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.val)


class ScalarSub(_ScalarMath):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sub(x, self.val)


class Concat(BaseModule):

    def __init__(self, dim: int = 0):
        super(Concat, self).__init__()
        self.dim = dim
        self.num_inputs = 2

    def forward(self, *inputs) -> torch.Tensor:
        self.num_inputs = len(inputs)
        return torch.cat(inputs, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Split(BaseModule):

    def __init__(self, split_size: int, dim: int = 0):
        super(Split, self).__init__()
        self.dim = dim
        self.split_size = split_size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Returns list of tensors, number of tensors = total_size // split_size.
        return torch.split(x, self.split_size, dim=self.dim)

    def extra_repr(self) -> str:
        return f"{self.split_size}, dim={self.dim}"


class Chunk(BaseModule):

    def __init__(self, chunks: int, dim: int = 0):
        super(Chunk, self).__init__()
        self.dim = dim
        self.chunks = chunks

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Returns list of tensors, number of tensors = chunks
        return torch.chunk(x, self.chunks, dim=self.dim)

    def extra_repr(self) -> str:
        return f"{self.chunks}, dim={self.dim}"


class Transpose(BaseModule):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        assert dim0 != dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, self.dim0, self.dim1)
