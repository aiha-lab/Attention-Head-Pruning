import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.modules.utils import to_pair_tuple, to_quadruple_tuple, to_n_tuple


class _ConstantPadNd(BaseModule):

    def __init__(self, value: float) -> None:
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, "constant", self.value)

    def extra_repr(self) -> str:
        return f"padding={self.padding}, value={self.value}"


class ConstantPad1d(_ConstantPadNd):

    def __init__(self, padding, value: float) -> None:
        super(ConstantPad1d, self).__init__(value=value, )
        self.padding = to_pair_tuple(padding)


class ConstantPad2d(_ConstantPadNd):

    def __init__(self, padding, value: float) -> None:
        super(ConstantPad2d, self).__init__(value=value, )
        self.padding = to_quadruple_tuple(padding)


class ConstantPad3d(_ConstantPadNd):

    def __init__(self, padding, value: float) -> None:
        super(ConstantPad3d, self).__init__(value=value, )
        self.padding = to_n_tuple(padding, 6)


class ZeroPad2d(ConstantPad2d):

    def __init__(self, padding) -> None:
        super(ZeroPad2d, self).__init__(padding, value=0.0, )


class _ReflectionPadNd(BaseModule):

    def __init__(self):
        super(_ReflectionPadNd, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, "reflect")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReflectionPad1d(_ReflectionPadNd):

    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__()
        self.padding = to_pair_tuple(padding)


class ReflectionPad2d(_ReflectionPadNd):

    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = to_quadruple_tuple(padding)


class ReflectionPad3d(_ReflectionPadNd):

    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = to_n_tuple(padding, 6)


class _ReplicationPadNd(BaseModule):

    def __init__(self):
        super(_ReplicationPadNd, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, "replicate")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReplicationPad1d(_ReplicationPadNd):

    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__()
        self.padding = to_pair_tuple(padding)


class ReplicationPad2d(_ReplicationPadNd):

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__()
        self.padding = to_quadruple_tuple(padding)


class ReplicationPad3d(_ReplicationPadNd):

    def __init__(self, padding):
        super(ReplicationPad3d, self).__init__()
        self.padding = to_n_tuple(padding, 6)


class CircularPad2d(BaseModule):

    def __init__(self, padding) -> None:
        super(CircularPad2d, self).__init__()
        self.padding = to_quadruple_tuple(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, "circular")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"
