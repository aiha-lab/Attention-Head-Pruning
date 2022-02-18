from typing import Optional, List, Tuple, Dict, Any
import math
import torch
import torch.nn as tnn

import nnlib.nn as nn
from nnlib.nn.transformer import SinusoidalPositionalEncoding
from .rel_all_transformer_layer import RelativeAllTransformerLayer


class AllTransformerLM(nn.BaseModule):

    def __init__(self, num_tokens: int, num_layers: int, hidden_dim: int, num_heads: int, feedforward_dim: int,
                 target_len: int, memory_len: int, extend_len: int = 0, max_len: Optional[int] = None,
                 attn_drop_prob: float = 0.1, proj_drop_prob: float = 0.1, qkv_drop_prob: float = 0.0,
                 attn_bias: bool = False, share_r_bias: bool = True, *,
                 input_drop_prob: Optional[float] = None, output_drop_prob: float = 0.0,
                 word_drop_prob: float = 0.0,
                 axial_drop: bool = False,
                 div_value: int = 1, tie_weight: bool = True, tie_proj: bool = True, cutoffs=(),
                 pos_clamp_length: Optional[int] = None, pre_norm: bool = True,
                 same_length: bool = True, norm_layer=None):
        super(AllTransformerLM, self).__init__()

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim

        self.adaptive = (len(cutoffs) > 0)
        if self.adaptive:
            self.word_emb = nn.AdaptiveEmbedding(num_tokens, hidden_dim, cutoffs, div_value=div_value,
                                                 word_drop_prob=word_drop_prob,
                                                 shortlist_proj=(div_value != 1))
        else:
            self.word_emb = nn.Embedding(num_tokens, hidden_dim, word_drop_prob=word_drop_prob)

        if input_drop_prob is None:
            input_drop_prob = proj_drop_prob
        if not axial_drop:
            self.drop_i = nn.Dropout(input_drop_prob)
            self.drop_o = nn.Dropout(output_drop_prob)
        else:
            self.drop_i = nn.AxialDropout(input_drop_prob, axis=1)  # seq axis
            self.drop_o = nn.AxialDropout(output_drop_prob, axis=1)  # seq axis
        self.drop = nn.Dropout(proj_drop_prob)

        self.target_len = target_len
        self.memory_len = memory_len
        self.extend_len = extend_len
        if max_len is None:
            max_len = target_len + memory_len + extend_len

        self.layers = nn.ModuleList([
            RelativeAllTransformerLayer(hidden_dim, num_heads, feedforward_dim, attn_drop_prob, proj_drop_prob,
                                        qkv_drop_prob,
                                        attn_bias=attn_bias, share_r_bias=share_r_bias,
                                        axial_drop=axial_drop,
                                        pre_norm=pre_norm, norm_layer=norm_layer)
            for _ in range(self.num_layers)])

        if self.adaptive:
            self.criterion = nn.AdaptiveLogSoftmaxWithLoss(num_tokens, hidden_dim, cutoffs, div_value=div_value,
                                                           bias=True, shortlist_proj=(div_value != 1))
        else:
            self.criterion = nn.LogSoftmaxWithLoss(num_tokens, hidden_dim, bias=True)

        # self.pos_emb = SinusoidalPositionalEncoding(hidden_dim, max_seq_length=max_len, clamp_length=pos_clamp_length,
        #                                             inverse=True)
        self.pos_emb = nn.ParameterModule(torch.zeros(1, max_len, self.hidden_dim))

        self.same_length = same_length

        self.share_r_bias = share_r_bias
        if share_r_bias:
            self.rel_query_bias = nn.ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
            self.rel_pos_bias = nn.ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
        else:
            self.rel_query_bias = None
            self.rel_pos_bias = None

        self._initialize_parameters()

        # Tie after initialization
        if self.adaptive:
            if tie_weight:  # tie embeddings
                self.criterion.shortlist[-1].weight = self.word_emb.shortlist[0].weight
                assert len(self.criterion.tail) == len(self.word_emb.tail)
                for i in range(len(self.criterion.tail)):
                    self.criterion.tail[i][-1].weight = self.word_emb.tail[i][0].weight
            if tie_proj and (div_value != 1):  # tie projections
                # do not tie shortlist weight?
                # transpose_flag = (self.criterion.shortlist[0].weight.shape != self.word_emb.shortlist[-1].weight.shape)
                # self.criterion.shortlist[0].weight = self.word_emb.shortlist[-1].weight
                # self.criterion.shortlist[0].weight_transposed = transpose_flag

                assert len(self.criterion.tail) == len(self.word_emb.tail)
                for i in range(len(self.criterion.tail)):
                    transpose_flag = (self.criterion.tail[i][0].weight.shape != self.word_emb.tail[i][-1].weight.shape)
                    self.criterion.tail[i][0].weight = self.word_emb.tail[i][-1].weight
                    self.criterion.tail[i][0].weight_transposed = transpose_flag
        else:
            # do not tie shortlist weight
            pass

        self.set_name()

    def _initialize_parameters(self):
        s1 = math.sqrt(1 / self.hidden_dim)
        s2 = math.sqrt(1 / self.hidden_dim / self.num_layers)
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                tnn.init.uniform_(module.weight.data, -s1, s1)
            elif isinstance(module, nn.Linear):
                if ('shortlist' in module_name) or ('tail' in module_name) or ('project' in module_name):
                    tnn.init.uniform_(module.weight.data, -s1, s1)
                else:
                    tnn.init.uniform_(module.weight.data, -s2, s2)
        for param_name, param in self.named_parameters():
            if ('mem_k' in param_name) or ('mem_v' in param_name):
                tnn.init.uniform_(param.data, -s2, s2)
            elif 'pos_emb' in param_name:
                tnn.init.uniform_(param.data, -s1, s1)

    def set_length(self, target_len: int, memory_len: int, extend_len: int = 0):
        self.target_len = target_len
        self.memory_len = memory_len
        self.extend_len = extend_len

    def set_same_length(self, flag: bool = True):
        self.same_length = flag

    def set_gating(self, flag: bool = True):
        for layer in self.layers:
            layer: RelativeAllTransformerLayer
            layer.set_gating(flag)

    def _init_memories(self, batch_size: int, dtype, device) -> List[torch.Tensor]:
        memory = []
        for i in range(self.num_layers + 1):
            memory.append(torch.zeros(batch_size, self.memory_len, self.hidden_dim, dtype=dtype, device=device))
        return memory

    def _update_memories(self, hiddens: List[torch.Tensor], prev_memories: Optional[List[torch.Tensor]],
                         memory_len: int, target_len: int) -> Optional[List[torch.Tensor]]:
        if prev_memories is not None:
            if len(hiddens) != len(prev_memories):
                raise ValueError(f'Memory length should be num_layers + 1, but got'
                                 f'num_layers: {self.num_layers}, '
                                 f'num_hiddens: {len(hiddens)} vs num_memories: {len(prev_memories)}.')

        # current: mem_len + tgt_len
        # target[-self.ext_len:] will be used as extended context
        # |-----mem-----|-----tgt-----|
        #   |-----new_mem-----|--ext--|
        with torch.no_grad():
            new_memories = []
            end_idx = memory_len + max(0, target_len - self.extend_len)
            begin_idx = max(0, end_idx - self.memory_len)
            for i in range(len(hiddens)):
                if prev_memories is not None:
                    m = torch.cat([prev_memories[i], hiddens[i]], dim=1)
                else:
                    m = hiddens[i]
                new_memories.append(m[:, begin_idx: end_idx].detach())
        return new_memories

    def forward(self, indices: torch.Tensor, target_indices: Optional[torch.Tensor] = None,
                     memories: Optional[List[torch.Tensor]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, int], Optional[List[torch.Tensor]]]:
        """
        indices:        (batch_size, target_length)
        target_indices: (batch_size, target_length)
        memories:       [ (batch_size, memory_length, hidden_dim) x (num_layers + 1384) ]
        """
        batch_size, target_len = indices.shape
        word_emb = self.word_emb(indices)  # (batch_size, target_len, hidden_dim)
        word_emb = self.drop_i(word_emb)
        word_emb = word_emb * (self.hidden_dim ** 0.5)  # to make same as Transformer-XL (why?)

        if memories is None:
            memories = self._init_memories(batch_size, word_emb.dtype, word_emb.device)

        memory_len = memories[0].shape[1] if (memories is not None) else 0
        source_len = target_len + memory_len

        attn_mask = torch.tril(
            torch.ones(target_len, source_len, dtype=torch.bool, device=word_emb.device),
            diagonal=memory_len)  # (target_len, source_len)
        if self.same_length:
            attn_mask = torch.logical_xor(attn_mask, torch.tril(
                torch.ones(target_len, source_len, dtype=torch.bool, device=word_emb.device),
                diagonal=-1))  # (target_len, source_len)

        # pos_emb = self.pos_emb(source_len)  # (1, source_len, hidden_dim)
        pos_emb = self.pos_emb()  # (1, max_len, head_dim)
        pos_emb = self.drop_i(pos_emb)

        h = word_emb
        hiddens = [h]

        gate_loss = 0
        sparsity_val = 0
        sparsity_count = 0

        for i, layer in enumerate(self.layers):
            memory_i = memories[i] if (memories is not None) else None

            if self.share_r_bias:
                r_query_bias = self.rel_query_bias()
                r_pos_bias = self.rel_pos_bias()
            else:
                r_query_bias = r_pos_bias = None

            h, layer_loss, layer_sparsity, _ = layer(h, pos_emb, r_query_bias, r_pos_bias, memory=memory_i,
                                                     attn_mask=attn_mask)
            hiddens.append(h)

            gate_loss = gate_loss + layer_loss
            sparsity_val = sparsity_val + layer_sparsity[0]
            sparsity_count = sparsity_count + layer_sparsity[1]

        h = self.drop(h)  # (batch_size, target_len, hidden_dim)
        new_memories = self._update_memories(hiddens, memories, memory_len, target_len)
        h = self.drop_o(h)

        if target_indices is not None:  # training
            if indices.shape[1] < target_indices.shape[1]:
                raise ValueError(f'Target length {target_indices.shape[1]} is longer than '
                                 f'input length {indices.shape[1]}.')
            elif indices.shape[1] > target_indices.shape[1]:
                h = h[:, -target_indices.shape[1]:]

            _, loss = self.criterion(h.view(-1, self.hidden_dim), target_indices.view(-1))
            return loss, gate_loss, (sparsity_val, sparsity_count), new_memories

        else:  # evaluation
            prob = self.criterion.log_prob(h.view(-1, self.hidden_dim))
            prob = prob.view(batch_size, target_len, self.num_tokens)
            return prob, gate_loss, (sparsity_val, sparsity_count), new_memories

    @classmethod
    def from_config(cls, config: Dict[str, Any], num_tokens: int,
                    target_len: int, memory_len: int, extend_len: int = 0):
        return cls(
            num_tokens=config.get('num_tokens', num_tokens),
            num_layers=config.get('num_layers'),
            hidden_dim=config.get('hidden_dim'),
            num_heads=config.get('num_heads'),
            feedforward_dim=config.get('feedforward_dim'),
            target_len=config.get('target_len', target_len),
            memory_len=config.get('memory_len', memory_len),
            extend_len=config.get('extend_len', extend_len),
            max_len=config.get('max_len', None),
            attn_drop_prob=config.get('attn_drop_prob', 0.0),
            proj_drop_prob=config.get('proj_drop_prob', 0.1),
            qkv_drop_prob=config.get('qkv_drop_prob', 0.0),
            input_drop_prob=config.get('input_drop_prob', None),
            output_drop_prob=config.get('output_drop_prob', 0.0),
            word_drop_prob=config.get('word_drop_prob', 0.0),
            axial_drop=config.get('axial_drop', False),
            attn_bias=config.get('attn_bias', False),
            share_r_bias=config.get('share_r_bias', True),
            div_value=config.get('div_value', 1),
            tie_weight=config.get('tie_weight', True),
            tie_proj=config.get('tie_proj', True),
            cutoffs=tuple(config.get('cutoffs', [])),
            pos_clamp_length=config.get('pos_clamp_length', None),
            pre_norm=config.get('pre_norm', True),
            same_length=config.get('same_length', True)
        )


"""Transformer-XL PPL

LM1B
- base:     0.46B   23.5
- large:    0.8B    21.8

Wikitext-103
- base:             24.03
- large:            18.03
"""
