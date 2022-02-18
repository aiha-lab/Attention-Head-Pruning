from typing import Optional, List, Tuple, Dict, Any
import math
import torch
import torch.nn as tnn

from nnlib.nn import (BaseModule, ModuleList, AdaptiveEmbedding, Embedding, AdaptiveLogSoftmaxWithLoss,
                      LogSoftmaxWithLoss, SequenceDropout, Linear, LayerNorm, ParameterModule)
from nnlib.nn.transformer import SinusoidalPositionalEncoding

from .rel_transformer_layer import RelativeTransformerLayer


class MemoryTransformer(BaseModule):

    def __init__(self,
                 num_tokens: int,
                 num_layers: int,
                 hidden_dim: int,
                 num_heads: int,
                 feedforward_dim: int,
                 seq_length: int,
                 mem_length: int,
                 overlap_length: int = 0,
                 attn_drop_prob: float = 0.1,
                 proj_drop_prob: float = 0.1,
                 qkv_drop_prob: float = 0.0,
                 feedforward_drop_prob: Optional[float] = None,
                 eps: float = 1e-5, *,
                 input_drop_prob: Optional[float] = None,
                 output_drop_prob: float = 0.0,
                 word_drop_prob: float = 0.0,
                 axial_drop: bool = False,
                 div_value: int = 1,
                 tie_weight: bool = True,
                 tie_proj: bool = True,
                 cutoffs=(),
                 pos_clamp_length: Optional[int] = None,
                 pre_norm: bool = False,
                 same_length: bool = False,
                 act_layer=None, norm_layer=None):
        super(MemoryTransformer, self).__init__()

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim

        self.adaptive = (len(cutoffs) > 0)
        if self.adaptive:
            self.word_emb = AdaptiveEmbedding(num_tokens, hidden_dim, cutoffs, div_value=div_value,
                                              padding_idx=None, shortlist_proj=(div_value != 1),
                                              word_drop_prob=word_drop_prob)
        else:
            self.word_emb = Embedding(num_tokens, hidden_dim, padding_idx=None, word_drop_prob=word_drop_prob)
        self.pos_emb = SinusoidalPositionalEncoding(hidden_dim, padding_idx=None,  # CRUCIAL!
                                                    clamp_length=pos_clamp_length, inverse=True)

        if input_drop_prob is None:
            input_drop_prob = proj_drop_prob

        self.use_axial_drop = axial_drop
        self.drop_i = SequenceDropout(input_drop_prob, axial_drop=axial_drop, axis=1)  # seq axis
        self.drop_o = SequenceDropout(output_drop_prob, axial_drop=axial_drop, axis=1)  # seq axis

        self.seq_length = seq_length
        self.mem_length = mem_length
        self.overlap_length = overlap_length
        self.same_length = same_length

        self.pre_norm = pre_norm
        self.layers = ModuleList([
            RelativeTransformerLayer(hidden_dim, num_heads, feedforward_dim, attn_drop_prob, proj_drop_prob,
                                     qkv_drop_prob, feedforward_drop_prob=feedforward_drop_prob, eps=eps,
                                     axial_drop=axial_drop, pre_norm=pre_norm,
                                     act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(self.num_layers)]
        )

        if pre_norm:
            self.out_norm = LayerNorm(hidden_dim, eps=eps)
        else:
            self.out_norm = None

        if self.adaptive:
            self.criterion = AdaptiveLogSoftmaxWithLoss(num_tokens, hidden_dim, cutoffs, div_value=div_value,
                                                        bias=True, shortlist_proj=(div_value != 1))
        else:
            self.criterion = LogSoftmaxWithLoss(num_tokens, hidden_dim, bias=True)

        self.query_key_bias = ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
        self.query_pos_bias = ParameterModule(torch.zeros(hidden_dim, dtype=torch.float32))
        self._initialize_parameters()

        # tie after initialization
        if self.adaptive:
            if tie_weight:  # tie embeddings
                self.criterion.shortlist[-1].weight = self.word_emb.shortlist[0].weight
                assert len(self.criterion.tail) == len(self.word_emb.tail)
                for i in range(len(self.criterion.tail)):
                    self.criterion.tail[i][-1].weight = self.word_emb.tail[i][0].weight
            if tie_proj and (div_value != 1):  # tie projections
                # do not tie shortlist weight
                assert len(self.criterion.tail) == len(self.word_emb.tail)
                for i in range(len(self.criterion.tail)):
                    transpose_flag = (self.criterion.tail[i][0].weight.shape != self.word_emb.tail[i][-1].weight.shape)
                    self.criterion.tail[i][0].weight = self.word_emb.tail[i][-1].weight
                    self.criterion.tail[i][0].weight_transposed = transpose_flag
        else:
            if tie_weight:
                self.criterion.project.weight = self.word_emb.weight

        self.set_name()

    # def _initialize_parameters(self):
    #     for module_name, module in self.named_modules():
    #         if isinstance(module, nn.Embedding):
    #             tnn.init.normal_(module.weight.data, 0.0, 0.01)
    #         elif isinstance(module, nn.Linear):
    #             if ("shortlist" in module_name) or ("tail" in module_name) or ("project" in module_name):
    #                 tnn.init.normal_(module.weight.data, 0.0, 0.01)
    #             else:
    #                 tnn.init.normal_(module.weight.data, 0.0, 0.02)
    #         elif isinstance(module, nn.LayerNorm):
    #             tnn.init.normal_(module.weight.data, 0.0, 0.02)  # ParameterModuleWithOffset
    #         else:
    #             continue
    #     if self.share_r_bias:
    #         tnn.init.normal_(self.rel_query_bias.data, 0.0, 0.02)
    #         tnn.init.normal_(self.rel_pos_bias.data, 0.0, 0.02)

    # def _initialize_parameters(self):
    #     s1 = math.sqrt(1 / self.hidden_dim)
    #     s2 = math.sqrt(1 / self.hidden_dim / self.num_layers)
    #     for module_name, module in self.named_modules():
    #         if isinstance(module, Embedding):
    #             tnn.init.uniform_(module.weight.data, -s1, s1)
    #         elif isinstance(module, Linear):
    #             if ("shortlist" in module_name) or ("tail" in module_name) or ("project" in module_name):
    #                 tnn.init.uniform_(module.weight.data, -s1, s1)
    #             else:
    #                 tnn.init.uniform_(module.weight.data, -s2, s2)

    def _initialize_parameters(self):
        # following http://proceedings.mlr.press/v119/huang20f/huang20f.pdf
        s = math.sqrt(1 / self.hidden_dim)
        s_ff = 0.67 * (self.num_layers ** (-0.25))
        s_v = s_ff * math.sqrt(2.0)
        for module_name, module in self.named_modules():
            if isinstance(module, Embedding):
                tnn.init.normal_(module.weight.data, 0, s)
            elif isinstance(module, Linear):
                if "attn" in module_name:
                    if "v_proj" in module_name:
                        tnn.init.xavier_uniform_(module.weight.data, gain=s_v)
                    else:  # q, k
                        tnn.init.xavier_uniform_(module.weight.data)
                else:  # proj, ff
                    tnn.init.xavier_uniform_(module.weight.data, gain=s_ff)

    def set_length(self, seq_length: int, mem_length: int, overlap_length: int = 0, same_length: bool = True):
        self.seq_length = seq_length
        self.mem_length = mem_length
        self.overlap_length = overlap_length
        self.same_length = same_length

    @torch.no_grad()
    def _init_memories(self, batch_size: int, dtype, device) -> List[torch.Tensor]:
        memory = []
        for i in range(self.num_layers + 1):
            memory.append(torch.zeros(batch_size, self.mem_length, self.hidden_dim, dtype=dtype, device=device))
        return memory

    @torch.no_grad()
    def _update_memories(self,
                         hiddens: List[torch.Tensor],
                         prev_memories: Optional[List[torch.Tensor]],
                         mem_length: int,
                         seq_length: int) -> Optional[List[torch.Tensor]]:
        if prev_memories is not None:
            if len(hiddens) != len(prev_memories):
                raise ValueError(f"[ERROR:MODEL] Memory length should be num_layers + 1, but got "
                                 f"num_layers: {self.num_layers}, "
                                 f"num_hiddens: {len(hiddens)} vs "
                                 f"num_memories: {len(prev_memories)}.")

        # current: mem_len + tgt_len
        # target[-self.ext_len:] will be used as extended context
        # |-------mem--------|-------seq--------|
        #   |-------new_mem-------|---overlap---|----|
        #                         |-----new seq------|
        new_memories = []
        end_idx = mem_length + max(0, seq_length - self.overlap_length)
        begin_idx = max(0, end_idx - self.mem_length)
        for i in range(len(hiddens)):
            if prev_memories is not None:
                m = torch.cat([prev_memories[i], hiddens[i]], dim=1)
            else:
                m = hiddens[i]
            new_memories.append(m[:, begin_idx:end_idx].clone().detach())
        return new_memories

    def forward(self,
                indices: torch.Tensor,
                target_indices: Optional[torch.Tensor] = None,
                memories: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        indices:            (batch_size, seq_length)
        target_indices:     (batch_size, seq_length)
        memories:           [ (batch_size, mem_length, hidden_dim) x (num_layers + 1) ]
        """
        batch_size, seq_length = indices.shape

        word_emb = self.word_emb(indices)  # (batch_size, seq_length, hidden_dim)
        word_emb = word_emb * (self.hidden_dim ** 0.5)
        word_emb = self.drop_i(word_emb)

        if memories is None:
            memories = self._init_memories(batch_size, word_emb.dtype, word_emb.device)

        mem_length = memories[0].shape[1] if (memories is not None) else 0
        source_len = seq_length + mem_length  # a.k.a. key_len

        _dummy_indices = torch.ones(1, source_len, dtype=torch.long, device=indices.device)
        pos_emb = self.pos_emb(_dummy_indices)  # (1, source_len, hidden_dim)
        pos_emb = self.drop_i(pos_emb)

        with torch.no_grad():
            # 1 1 1 1 0 0
            # 1 1 1 1 1 0
            # 1 1 1 1 1 1
            attn_mask = torch.tril(
                torch.ones(seq_length, source_len, dtype=torch.bool, device=word_emb.device),
                diagonal=mem_length)  # (seq_length, source_len)
            if self.same_length:
                # 1 1 1 1 0 0   1 1 1 1 0 0     0 0 0 0 0 0
                # 0 1 1 1 1 0 = 1 1 1 1 1 0 xor 1 0 0 0 0 0
                # 0 0 1 1 1 1   1 1 1 1 1 1     1 1 0 0 0 0
                attn_mask = torch.logical_xor(
                    attn_mask, torch.tril(
                        torch.ones(seq_length, source_len, dtype=torch.bool, device=word_emb.device), diagonal=-1
                    )
                )  # (seq_length, source_len)

        h = word_emb
        hiddens = [h]

        query_key_bias = self.query_key_bias()
        query_pos_bias = self.query_pos_bias()

        for i, layer in enumerate(self.layers):
            memory_i = memories[i] if (memories is not None) else None

            h, _ = layer(h, pos_emb, query_key_bias, query_pos_bias, memory=memory_i, attn_mask=attn_mask)
            hiddens.append(h.detach())

        if self.out_norm is not None:
            h = self.out_norm(h)
        h = self.drop_o(h)

        new_memories = self._update_memories(hiddens, memories, mem_length, seq_length)

        if target_indices is not None:  # training
            if indices.shape[1] < target_indices.shape[1]:
                raise ValueError(f"[ERROR:MODEL] Target length {target_indices.shape[1]} is longer than "
                                 f"input length {indices.shape[1]}.")
            elif indices.shape[1] > target_indices.shape[1]:
                h = h[:, -target_indices.shape[1]:]

            _, loss = self.criterion(h.view(-1, self.hidden_dim), target_indices.view(-1))
            return loss, new_memories

        else:  # evaluation
            prob = self.criterion.log_prob(h.view(-1, self.hidden_dim))
            prob = prob.view(batch_size, seq_length, self.num_tokens)
            return prob, new_memories

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    num_tokens: int,
                    seq_length: int,
                    mem_length: int,
                    overlap_length: int = 0, **kwargs):
        return cls(
            num_tokens=config.get("num_tokens", num_tokens),
            num_layers=config.get("num_layers"),
            hidden_dim=config.get("hidden_dim"),
            num_heads=config.get("num_heads"),
            feedforward_dim=config.get("feedforward_dim"),
            seq_length=config.get("seq_length", seq_length),
            mem_length=config.get("mem_length", mem_length),
            overlap_length=config.get("overlap_length", overlap_length),
            attn_drop_prob=config.get("attn_drop_prob", 0.0),
            proj_drop_prob=config.get("proj_drop_prob", 0.1),
            qkv_drop_prob=config.get("qkv_drop_prob", 0.0),
            feedforward_drop_prob=config.get("feedforward_drop_prob", None),
            eps=config.get("eps", 1e-5),
            input_drop_prob=config.get("input_drop_prob", None),
            output_drop_prob=config.get("output_drop_prob", 0.0),
            word_drop_prob=config.get("word_drop_prob", 0.0),
            axial_drop=config.get("axial_drop", False),
            div_value=config.get("div_value", 1),
            tie_weight=config.get("tie_weight", True),
            tie_proj=config.get("tie_proj", True),
            cutoffs=tuple(config.get("cutoffs", [])),
            pos_clamp_length=config.get("pos_clamp_length", None),
            pre_norm=config.get("pre_norm", False),
            same_length=config.get("same_length", False)
        )
