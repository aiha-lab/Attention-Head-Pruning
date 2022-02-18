from typing import Union, Tuple
import torch

from nnlib.nn.modules.module import BaseModule


class Flatten(BaseModule):

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(BaseModule):

    def __init__(self, dim: Union[int, str], unflattened_size: [torch.Size, Tuple]) -> None:
        super(Unflatten, self).__init__()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)
        else:
            raise TypeError("[ERROR:NN] invalid argument type for dim parameter")

        self.dim = dim
        self.unflattened_size = unflattened_size

    @staticmethod
    def _require_tuple_tuple(input_size):
        if isinstance(input_size, tuple):
            for idx, elem in enumerate(input_size):
                if not isinstance(elem, tuple):
                    raise TypeError("[ERROR:NN] unflattened_size must be tuple of tuples, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("[ERROR:NN} unflattened_size must be a tuple of tuples, " +
                        "but found type {}".format(type(input_size).__name__))

    @staticmethod
    def _require_tuple_int(input_size):
        if isinstance(input_size, tuple):
            for idx, elem in enumerate(input_size):
                if not isinstance(elem, int):
                    raise TypeError("[ERROR:NN] unflattened_size must be tuple of ints, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError(
            "[ERROR:NN] unflattened_size must be a tuple of ints, but found type {}".format(type(input_size).__name__))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"
