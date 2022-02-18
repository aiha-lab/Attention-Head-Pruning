import torch

from nnlib.nn.modules import BaseModule, GlobalAvgPool2d, Conv2d, Mul, ReLU, Sigmoid


class SqueezeExcite(BaseModule):

    def __init__(self,
                 in_channels: int,
                 reduced_channels: int, *,
                 act_layer=None) -> None:
        super(SqueezeExcite, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels
        if act_layer is not None:
            act_layer = ReLU
        self.act = act_layer()
        if hasattr(self.act, "inplace"):
            self.act.inplace = True

        self.pool = GlobalAvgPool2d(keepdim=True)
        self.fc0 = Conv2d(in_channels, reduced_channels, 1, 1, 0, bias=True)
        self.fc1 = Conv2d(reduced_channels, in_channels, 1, 1, 0, bias=True)
        self.sigmoid = Sigmoid()
        self.mul = Mul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"[ERROR:NN] SqueezeExcite input should be 4D tensor, got {x.shape}.")

        y = self.pool(x)  # (n, c, 1, 1)
        y = self.act(self.fc0(y))
        y = self.sigmoid(self.fc1(y))
        return self.mul(x, y)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.reduced_channels}"
