from typing import Optional, Tuple
import math
import torch

import happy_torch.nn as nn


class SimpleGating(nn.BaseModule):

    def __init__(self, num_gates: int, init: float = 0.0, beta: float = 0.667, gamma: float = -0.1, zeta: float = 1.1,
                 hard: bool = True, *, name: Optional[str] = None):
        super(SimpleGating, self).__init__()
        self.num_gates = num_gates
        self.beta = beta

        assert (gamma <= 0) and (zeta >= 1)
        self.gamma = gamma
        self.zeta = zeta

        self.gate = nn.ParameterModule(torch.zeros(num_gates, ).fill_(init))
        self.noise = nn.BufferModule(torch.zeros(num_gates, ), persistent=False)
        self.threshold = 0.5
        self.eps = 1e-6
        self.loss_constant = (beta * math.log(-gamma / zeta)) if (gamma < 0) else 0.0
        self.hard = hard

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, int]]:
        if self.training:
            self.noise.data.uniform_(self.eps, 1 - self.eps)  # random generate
            u = self.noise.data

            s = torch.log(u) - torch.log(1.0 - u)
            s = (s + self.gate()) / self.beta
            s = torch.sigmoid(s)
        else:
            s = torch.sigmoid(self.gate())
        s = s * (self.zeta - self.gamma) + self.gamma
        out = torch.clamp(s, self.eps, 1)

        if self.hard:
            out_hard = torch.greater_equal(out, self.threshold).float()
            out = out + (out_hard - out).detach()
            sparsity = torch.eq(out, 0).sum()
        else:
            # TODO different to train-test
            sparsity = torch.less(out, self.threshold).float().sum()

        l0_loss = torch.sigmoid(self.gate() - self.loss_constant)
        l0_loss = torch.clamp(l0_loss, self.eps, 1.0 - self.eps).sum()
        # l2_loss = torch.sigmoid(self.gate() - self.loss_constant) * self.gate() * self.gate()
        # l2_loss = l2_loss.sum()
        # loss = l0_loss + 1e-4 * l2_loss
        loss = l0_loss
        return out, loss, (sparsity, self.num_gates)


if __name__ == '__main__':
    g = SimpleGating(8)
    g.train()
    print(g())
    print(g())
    print(g())

    g.eval()
    print(g())
    print(g())
    print(g())
