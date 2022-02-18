from typing import Optional, Tuple
import torch
import torch.nn as tnn
import torch.nn.functional as F

from nnlib.nn.parameter import ParameterModule
from nnlib.nn.modules.normalization import LayerNorm
from nnlib.nn.modules.module import _BaseModuleMixin, BaseModule


class LSTM(tnn.LSTM, _BaseModuleMixin):
    # https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
    # WARN: parameters are not Modularized with ParameterModule.

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 *, init_forget_bias: bool = True,
                 variational_dropout: bool = False) -> None:
        tnn.LSTM.__init__(self, input_dim, hidden_dim, num_layers, bias, batch_first,
                          dropout if (not variational_dropout) else 0.0, bidirectional)
        _BaseModuleMixin.__init__(self)

        self.variational_dropout = dropout if variational_dropout else 0.0  # value
        self._initialize_parameters(init_forget_bias)

    def _initialize_parameters(self, init_forget_bias: bool = True):
        for p_name, p in self.named_parameters():
            if "weight_hh" in p_name:
                tnn.init.orthogonal_(p.data)
            elif "weight_ih" in p_name:
                tnn.init.xavier_uniform_(p.data)
            elif ("bias" in p_name) and init_forget_bias:
                tnn.init.zeros_(p.data)
                p.data[self.hidden_size:self.hidden_size * 2] = 1.0  # forget gate

    def _variational_drop_weight(self):
        for p_name, p in self.named_parameters():
            if "weight_hh" in p_name:
                getattr(self, p_name).data = F.dropout(p.data, p=self.variational_dropout,
                                                       training=self.training).contiguous()

    def forward(self,
                x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._variational_drop_weight()
        out, state = super().forward(x, hx)
        return out, state


class LSTMCell(BaseModule):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 dropout: float = 0.0,
                 *, init_forget_bias: bool = True,
                 variational_dropout: bool = False):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = bias
        self.dropout = dropout
        self.variational_dropout = dropout if variational_dropout else 0.0  # value

        self.weight_ih = ParameterModule(torch.zeros(4 * hidden_dim, input_dim))
        self.weight_hh = ParameterModule(torch.zeros(4 * hidden_dim, hidden_dim))
        if bias:
            self.bias_ih = ParameterModule(torch.zeros(4 * hidden_dim))
            self.bias_hh = ParameterModule(torch.zeros(4 * hidden_dim))
        else:
            self.bias_ih = self.bias_hh = None

        self._initialize_parameters(init_forget_bias)

    def _initialize_parameters(self, init_forget_bias: bool = True):
        for p_name, p in self.named_parameters():
            if "weight_hh" in p_name:
                tnn.init.orthogonal_(p.data)
            elif "weight_ih" in p_name:
                tnn.init.xavier_uniform_(p.data)
            elif ("bias" in p_name) and init_forget_bias:
                tnn.init.zeros_(p.data)
                p.data[self.hidden_dim:self.hidden_dim * 2] = 1.0  # forget gate

    def _generate_zero_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        zeros = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        state = (zeros, zeros)
        return state

    def _check_state(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(state) != 2:
            raise ValueError(f"[ERROR] LSTMCell state should be tuple of two tensors.")

        batch_size = x.shape[0]
        if state[0].shape != (batch_size, self.hidden_dim):
            raise ValueError(f"[ERROR] LSTMCell state[0] shape {state[0].shape}, "
                             f"should be ({batch_size}, {self.hidden_dim}).")
        if state[1].shape != (batch_size, self.hidden_dim):
            raise ValueError(f"[ERROR] LSTMCell state[1] shape {state[1].shape}, "
                             f"should be ({batch_size}, {self.hidden_dim}).")

    def lstm_core(self,
                  x: torch.Tensor,
                  weight_ih: torch.Tensor,
                  weight_hh: torch.Tensor,
                  bias_ih: Optional[torch.Tensor],
                  bias_hh: Optional[torch.Tensor],
                  state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, cx = state

        # gate order: [in, forget, cell, out]
        ih_gates = torch.mm(x, weight_ih.transpose(0, 1))
        hh_gates = torch.mm(hx, weight_hh.transpose(0, 1))
        gates = ih_gates + hh_gates
        if self.use_bias:
            gates = gates + (bias_hh + bias_ih)

        in_gate, forget_gate, cell_gate, out_gate = torch.chunk(gates, 4, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)
        return hy, cy

    def forward(self,
                x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x:      (batch_size, hidden_dim)
        state: (batch_size, hidden_dim), (batch_size, hidden_dim)
        """
        if state is None:
            state = self._generate_zero_state(x)
        else:
            self._check_state(x, state)

        weight_ih = self.weight_ih()
        weight_hh = F.dropout(self.weight_hh(), self.variational_dropout, training=self.training)  # variational drop

        bias_ih = self.bias_ih() if (self.bias_ih is not None) else None
        bias_hh = self.bias_hh() if (self.bias_hh is not None) else None

        hy, cy = self.lstm_core(x, weight_ih, weight_hh, bias_ih, bias_hh, state)

        new_state = (hy, cy)
        if self.dropout > 0 and (self.variational_dropout == 0):  # output drop
            hy = F.dropout(hy, self.dropout, training=self.training)

        return hy, new_state


class LayerNormLSTMCell(LSTMCell):
    """Following https://arxiv.org/pdf/1909.12415.pdf,
    but for efficiency, LN is merged, so should be slightly different.
    More similar to https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 dropout: float = 0.0,
                 *, eps: float = 1e-5,
                 init_forget_bias: bool = True,
                 variational_dropout: bool = False):
        super(LayerNormLSTMCell, self).__init__(input_dim, hidden_dim, bias, dropout,
                                                init_forget_bias=init_forget_bias,
                                                variational_dropout=variational_dropout)
        self.ih_norm = LayerNorm(4 * self.hidden_dim, eps=eps)
        self.hh_norm = LayerNorm(4 * self.hidden_dim, eps=eps)
        self.candidate_norm = LayerNorm(self.hidden_dim, eps=eps)

    def lstm_core(self,
                  x: torch.Tensor,
                  weight_ih: torch.Tensor,
                  weight_hh: torch.Tensor,
                  bias_ih: Optional[torch.Tensor],
                  bias_hh: Optional[torch.Tensor],
                  state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, cx = state

        # gate order: [in, forget, cell, out]
        ih_gates = self.ih_norm(torch.mm(x, weight_ih.transpose(0, 1)))
        hh_gates = self.hh_norm(torch.mm(hx, weight_hh.transpose(0, 1)))
        gates = ih_gates + hh_gates
        if self.use_bias:
            gates = gates + (bias_hh + bias_ih)

        in_gate, forget_gate, cell_gate, out_gate = torch.chunk(gates, 4, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(self.candidate_norm(cy))
        return hy, cy


def scan_lstm_cell(cell,
                   x: torch.Tensor,
                   state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   reverse: bool = False,
                   ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    x:      (batch_size, seq_length, input_dim)
    state:  [(batch_size, hidden_dim), (batch_size, hidden_dim)]
    """
    seq_length = x.shape[1]
    result = []

    x = x.transpose(0, 1)  # (s, b, d)

    if not reverse:
        for i in range(seq_length):
            h, state = cell(x[i], state)
            result.append(h)
    else:
        for i in range(seq_length - 1, -1, -1):
            h, state = cell(x[i], state)
            result.append(h)
        result = list(reversed(result))

    result = torch.stack(result, dim=1)  # (b, s, d)
    return result, state
