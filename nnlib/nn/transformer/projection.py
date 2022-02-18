import torch

from nnlib.nn.modules import BaseModule, Linear, SequenceDropout, Add, LayerNorm


class ProjectionResidualNorm(BaseModule):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 drop_prob: float = 0.0,
                 bias: bool = True,
                 eps: float = 1e-5, *,
                 axial_drop: bool = False,
                 norm_layer=None):
        super(ProjectionResidualNorm, self).__init__()
        if norm_layer is None:
            norm_layer = LayerNorm

        self.linear = Linear(in_dim, out_dim, bias=bias)
        self.use_axial_drop = axial_drop
        self.drop = SequenceDropout(drop_prob, axial_drop=axial_drop, axis=1, inplace=True)  # seq axis
        self.add = Add()
        self.norm = norm_layer(out_dim, eps=eps)

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        hidden:     (batch_size, sequence_length, in_dim)
        identity:   (batch_size, sequence_length, out_dim)
        """
        hidden = self.linear(hidden)
        hidden = self.drop(hidden)
        out = self.add(hidden, identity)
        out = self.norm(out)
        return out

    def extra_repr(self) -> str:
        s = f"{self.linear.in_features}, {self.linear.out_features}"
        if self.drop.p > 0:
            s += f", drop_prob={self.drop.p}"
            if self.use_axial_drop:
                s += f", axial_drop=True"
        return s


class ProjectionResidual(BaseModule):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 drop_prob: float = 0.0,
                 bias: bool = True, *,
                 axial_drop: bool = False):
        super(ProjectionResidual, self).__init__()

        self.linear = Linear(in_dim, out_dim, bias=bias)
        self.use_axial_drop = axial_drop
        self.drop = SequenceDropout(drop_prob, axial_drop=axial_drop, axis=1, inplace=True)  # seq axis
        self.add = Add()

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        hidden:     (batch_size, sequence_length, in_dim)
        identity:   (batch_size, sequence_length, out_dim)
        """
        hidden = self.linear(hidden)
        hidden = self.drop(hidden)
        out = self.add(hidden, identity)
        return out

    def extra_repr(self) -> str:
        s = f"{self.linear.in_features}, {self.linear.out_features}"
        if self.drop.p > 0:
            s += f", drop_prob={self.drop.p}"
            if self.use_axial_drop:
                s += f", axial_drop=True"
        return s
