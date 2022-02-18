import torch

from nnlib.nn.modules.module import BaseModule


class ChannelShuffle(BaseModule):

    def __init__(self, groups: int) -> None:
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("[ERROR:NN] ChannelShuffle input should be 4D tensor.")
        b, ch, h, w = x.shape
        if ch % self.groups != 0:
            raise ValueError("[ERROR:NN] ChannelShuffle input channel should be divisible by groups")
        ch_per_g = ch // self.groups

        x = x.view(b, self.groups, ch_per_g, h, w)
        x = torch.transpose(x, 1, 2).contiguous().view(b, -1, h, w)
        return x

    def extra_repr(self) -> str:
        return f"groups={self.groups}"
