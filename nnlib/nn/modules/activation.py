from typing import Optional
import torch
import torch.nn.functional as F

from nnlib.nn.modules.module import BaseModule
from nnlib.nn.parameter import ParameterModule
from nnlib.nn.functional import pact


class Threshold(BaseModule):

    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.threshold(x, self.threshold, self.value, self.inplace)

    def extra_repr(self) -> str:
        s = f"threshold={self.threshold}, value={self.value}"
        return s + (f"inplace=True" if self.inplace else "")


class ReLU(BaseModule):

    def __init__(self, inplace: bool = False, *, act_fn=None):
        super(ReLU, self).__init__()
        self.inplace = inplace
        if act_fn is None:
            act_fn = F.relu
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class LeakyReLU(BaseModule):

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False, *, act_fn=None):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        if act_fn is None:
            act_fn = F.leaky_relu
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x, self.negative_slope, inplace=self.inplace)

    def extra_repr(self) -> str:
        s = f"{self.negative_slope}"
        return s + ("inplace=True" if self.inplace else "")


class RReLU(BaseModule):

    def __init__(self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rrelu(x, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self) -> str:
        s = f"lower={self.lower}, upper={self.upper}"
        return s + ("inplace=True" if self.inplace else "")


class PReLU(BaseModule):

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = ParameterModule(torch.empty(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.weight())

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


class HardTanh(BaseModule):

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False, *, act_fn=None):
        super(HardTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        if act_fn is None:
            act_fn = F.hardtanh
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        s = f"min_val={self.min_val}, max_val={self.max_val}"
        return s + ("inplace=True" if self.inplace else "")


class ReLU6(HardTanh):

    def __init__(self, inplace: bool = False, *, act_fn=None):
        super(ReLU6, self).__init__(0, 6, inplace, act_fn=act_fn)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class GLU(BaseModule):

    def __init__(self, dim: int = -1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class _ELUFunc(BaseModule):

    def __init__(self, alpha: float = 1.0, inplace: bool = False, *, elu_fn=None):
        super(_ELUFunc, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.elu_fn = elu_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_fn(x, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        s = f"alpha={self.alpha}"
        return s + (f"inplace=True" if self.inplace else "")


class ELU(_ELUFunc):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super(ELU, self).__init__(alpha, inplace=inplace, elu_fn=F.elu)


class CELU(_ELUFunc):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super(CELU, self).__init__(alpha, inplace=inplace, elu_fn=F.celu)


class _NoInplaceAct(BaseModule):

    def __init__(self, *, act_fn=None):
        super(_NoInplaceAct, self).__init__()
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x)


class Sigmoid(_NoInplaceAct):

    def __init__(self):
        super(Sigmoid, self).__init__(act_fn=torch.sigmoid)


class Tanh(_NoInplaceAct):

    def __init__(self):
        super(Tanh, self).__init__(act_fn=torch.tanh)


class GELU(_NoInplaceAct):

    def __init__(self):
        super(GELU, self).__init__(act_fn=F.gelu)


class Softsign(_NoInplaceAct):

    def __init__(self):
        super(Softsign, self).__init__(act_fn=F.softsign)


class LogSigmoid(_NoInplaceAct):

    def __init__(self):
        super(LogSigmoid, self).__init__(act_fn=F.logsigmoid)


class Tanhshrink(_NoInplaceAct):

    def __init__(self):
        super(Tanhshrink, self).__init__(act_fn=F.tanhshrink)


class _LambdFunc(BaseModule):

    def __init__(self, lambd: float = 0.5, *, lambd_fn=None):
        super(_LambdFunc, self).__init__()
        self.lambd = lambd
        self.lambd_fn = lambd_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd_fn(x, self.lambd)

    def extra_repr(self) -> str:
        return f"{self.lambd}"


class Hardshrink(_LambdFunc):

    def __init__(self, lambd: float = 0.5):
        super(Hardshrink, self).__init__(lambd, lambd_fn=F.hardshrink)


class Softshrink(_LambdFunc):

    def __init__(self, lambd: float = 0.5):
        super(Softshrink, self).__init__(lambd, lambd_fn=F.softshrink)


class _InplaceAct(BaseModule):

    def __init__(self, inplace: bool = False, *, act_fn=None):
        super(_InplaceAct, self).__init__()
        self.inplace = inplace
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class HardSigmoid(_InplaceAct):

    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__(inplace=inplace, act_fn=F.hardsigmoid)


class SiLU(_InplaceAct):

    def __init__(self, inplace: bool = False):
        super(SiLU, self).__init__(inplace=inplace, act_fn=F.silu)


Swish = SiLU


class HardSwish(_InplaceAct):

    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__(inplace=inplace, act_fn=F.hardswish)


class SELU(_InplaceAct):
    def __init__(self, inplace: bool = False):
        super(SELU, self).__init__(inplace=inplace, act_fn=F.selu)


class Identity(BaseModule):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Softplus(BaseModule):

    def __init__(self, beta: int = 1, threshold: int = 20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


class _ExpFunc(BaseModule):

    def __init__(self, dim: Optional[int] = None, exp_fn=None):
        super(_ExpFunc, self).__init__()
        self.dim = dim
        self.exp_fn = exp_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.exp_fn(x, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Softmin(_ExpFunc):

    def __init__(self, dim: Optional[int] = None):
        super(Softmin, self).__init__(dim, exp_fn=F.softmin)


class Softmax(_ExpFunc):

    def __init__(self, dim: Optional[int] = None):
        super(Softmax, self).__init__(dim, exp_fn=F.softmax)


class LogSoftmax(_ExpFunc):

    def __init__(self, dim: Optional[int] = None):
        super(LogSoftmax, self).__init__(dim, exp_fn=F.log_softmax)


class Softmax2d(BaseModule):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Softmax2d input should be 4D.")
        return F.softmax(x, dim=1, _stacklevel=5)


class PACT(BaseModule):

    def __init__(self, init: float = 10):
        super(PACT, self).__init__()
        self.alpha = ParameterModule(torch.ones(1).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pact(x, self.alpha())


def get_activation_fn(name_or_cls):
    """Get activation function by name. Only available for non-parametric functions."""
    if isinstance(name_or_cls, str):
        name = name_or_cls.lower()
        if name == 'relu':
            return ReLU
        if name == 'leaky_relu' or name == 'lrelu':
            return LeakyReLU
        elif name == 'relu6':
            return ReLU6
        elif name == 'glu':
            return GLU
        elif name == 'elu':
            return ELU
        elif name == 'sigmoid':
            return Sigmoid
        elif name == 'tanh':
            return Tanh
        elif name == 'silu' or name == 'swish':
            return SiLU
        elif name == 'gelu':
            return GELU
        elif name == 'hardsigmoid' or name == 'hard_sigmoid':
            return HardSigmoid
        elif name == 'hardswish' or name == 'hard_swish':
            return HardSwish
        elif name == 'hardtanh' or name == 'hard_tanh':
            return HardTanh
        elif name == 'id' or name == 'identity':
            return Identity
        else:
            raise ValueError(f"[ERROR:NN] Invalid activation call, class name: {name}")
    elif isinstance(name_or_cls, BaseModule):
        return name_or_cls
    else:
        raise ValueError(f"[ERROR:NN] Invalid input type {type(name_or_cls)}, expected str or BaseModule.")
