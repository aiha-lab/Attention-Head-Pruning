from typing import Optional
import torch

from nnlib.nn.modules import BaseModule, Linear, ReLU, SequenceDropout, Add, LayerNorm


class _FeedForwardResidualBase(BaseModule):

    def __init__(self,
                 in_dim: int,
                 feedforward_dim: int,
                 drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None,
                 eps: float = 1e-5, *,
                 add_weight: float = 1.0,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(_FeedForwardResidualBase, self).__init__()

        if act_layer is None:
            act_layer = ReLU
        if feedforward_drop_prob is None:
            feedforward_drop_prob = drop_prob
        if norm_layer is None:
            norm_layer = LayerNorm

        self.linear1 = Linear(in_dim, feedforward_dim)
        self.linear2 = Linear(feedforward_dim, in_dim)
        self.act = act_layer()
        if hasattr(self.act, "inplace"):
            self.act.inplace = False
        self.norm = norm_layer(in_dim, eps=eps)

        self.use_axial_drop = axial_drop
        self.drop1 = SequenceDropout(feedforward_drop_prob, axial_drop=axial_drop, axis=1, inplace=True)  # seq axis
        self.drop2 = SequenceDropout(drop_prob, axial_drop=axial_drop, axis=1, inplace=True)  # seq axis

        self.add = Add()
        self.add_weight = add_weight

    def forward(self, hidden: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = f"{self.linear1.in_features}, feedforward_dim={self.linear1.out_features}"
        if self.drop2.p > 0:
            s += f", drop_prob={self.drop2.p}"
        if self.drop1.p > 0:
            s += f", feedforward_drop_prob={self.drop1.p}"
        if self.use_axial_drop:
            s += f", axial_drop=True"
        if self.add_weight != 1.0:
            s += f", add_weight={self.add_weight}"
        return s


class FeedForwardResidualNorm(_FeedForwardResidualBase):

    def __init__(self,
                 in_dim: int,
                 feedforward_dim: int,
                 drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None,
                 eps: float = 1e-5, *,
                 add_weight: float = 1.0,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(FeedForwardResidualNorm, self).__init__(in_dim, feedforward_dim, drop_prob, feedforward_drop_prob, eps,
                                                      add_weight=add_weight, axial_drop=axial_drop,
                                                      act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, hidden: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden:     (batch_size, sequence_length, in_dim)
        identity:   (batch_size, sequence_length, in_dim)
        """
        if identity is None:
            identity = hidden
        hidden = self.linear1(hidden)
        hidden = self.drop1(hidden)
        hidden = self.act(hidden)

        hidden = self.linear2(hidden)
        hidden = self.drop2(hidden)

        if self.add_weight != 1.0:
            out = self.add(hidden * self.add_weight, identity)
        else:
            out = self.add(hidden, identity)
        out = self.norm(out)
        return out


class NormFeedForwardResidual(_FeedForwardResidualBase):

    def __init__(self,
                 in_dim: int,
                 feedforward_dim: int,
                 drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None,
                 eps: float = 1e-5, *,
                 add_weight: float = 1.0,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(NormFeedForwardResidual, self).__init__(in_dim, feedforward_dim, drop_prob, feedforward_drop_prob, eps,
                                                      add_weight=add_weight, axial_drop=axial_drop,
                                                      act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, hidden: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden:     (batch_size, sequence_length, in_dim)
        identity:   (batch_size, sequence_length, in_dim)
        """
        if identity is None:
            identity = hidden

        hidden = self.norm(hidden)
        hidden = self.linear1(hidden)
        hidden = self.drop1(hidden)
        hidden = self.act(hidden)

        hidden = self.linear2(hidden)
        hidden = self.drop2(hidden)

        if self.add_weight != 0:
            out = self.add(hidden * self.add_weight, identity)
        else:
            out = self.add(hidden, identity)
        return out


class FeedForwardResidual(_FeedForwardResidualBase):

    def __init__(self,
                 in_dim: int,
                 feedforward_dim: int,
                 drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None,
                 eps: float = 1e-5, *,
                 add_weight: float = 1.0,
                 axial_drop: bool = False,
                 act_layer=None,
                 norm_layer=None):
        super(FeedForwardResidual, self).__init__(in_dim, feedforward_dim, drop_prob, feedforward_drop_prob, eps,
                                                  add_weight=add_weight, axial_drop=axial_drop,
                                                  act_layer=act_layer, norm_layer=norm_layer)
        del self.norm  # no LN inside

    def forward(self, hidden: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden:     (batch_size, sequence_length, in_dim)
        identity:   (batch_size, sequence_length, in_dim)
        """
        if identity is None:
            identity = hidden
        hidden = self.linear1(hidden)
        hidden = self.drop1(hidden)
        hidden = self.act(hidden)

        hidden = self.linear2(hidden)
        hidden = self.drop2(hidden)

        if self.add_weight != 1.0:
            out = self.add(hidden * self.add_weight, identity)
        else:
            out = self.add(hidden, identity)
        return out
