from typing import Optional, Sequence, Tuple, List
import torch
import torch.nn.functional as F

from nnlib.nn.modules import BaseModule, Linear, Embedding, Sequential, ModuleList


def _preprocess_cutoffs(cutoffs: Sequence[int], num_classes: int) -> List[int]:
    cutoffs = sorted(list(cutoffs))

    if len(cutoffs) > 0:
        if cutoffs[-1] == num_classes:
            cutoffs = cutoffs[:-1]
        if cutoffs[0] == 0:
            cutoffs = cutoffs[1:]

    if len(cutoffs) > 0:
        if (cutoffs != sorted(cutoffs)) or \
                (min(cutoffs) < 0) or \
                (max(cutoffs) >= num_classes) or \
                (len(set(cutoffs)) != len(cutoffs)) or \
                any([int(c) != c for c in cutoffs]):
            raise ValueError(
                "[ERROR:NN] AdaptiveLogSoftmaxWithLoss cutoff should be a sequence of unique, positive integers "
                "sorted in an increasing order, each value is between [1, num_classes).")
    else:  # len(cutoffs) == 0:
        raise ValueError("[ERROR:NN] Cutoffs empty. Consider using native softmax")

    return cutoffs


class AdaptiveLogSoftmaxWithLoss(BaseModule):

    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 cutoffs: Sequence[int],
                 div_value: float = 4.0,
                 bias: bool = True,
                 shortlist_proj: bool = False,
                 reduction: str = "mean") -> None:
        super(AdaptiveLogSoftmaxWithLoss, self).__init__()

        cutoffs = _preprocess_cutoffs(cutoffs, num_classes)

        self.num_features = num_features
        self.num_classes = num_classes

        self.cutoffs = cutoffs + [num_classes]  # [1000, 5000,] + [10000]: last index of each
        assert len(self.cutoffs) >= 2
        self.div_value = div_value
        self.use_bias = bias
        self.use_shortlist_proj = shortlist_proj
        self.reduction = reduction

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1

        if not shortlist_proj:
            self.shortlist = Sequential(
                Linear(self.num_features, self.shortlist_size, bias=bias),
            )  # for consistency (nn.Sequential)
        else:
            self.shortlist = Sequential(
                Linear(self.num_features, self.num_features, bias=False),
                Linear(self.num_features, self.shortlist_size, bias=bias)  # embedding
            )
        self.cluster = Linear(self.num_features, self.n_clusters, bias=bias)

        self.tail = ModuleList()
        for i in range(self.n_clusters):
            out_size = self.cutoffs[i + 1] - self.cutoffs[i]
            if div_value == 1:
                self.tail.append(Sequential(
                    Linear(self.num_features, out_size, bias=bias),  # embedding
                ))  # for consistency (nn.Sequential)
            else:
                head_size = int(num_features / (self.div_value ** (i + 1)))
                self.tail.append(Sequential(
                    Linear(self.num_features, head_size, bias=False),
                    Linear(head_size, out_size, bias=bias)  # embedding
                ))

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] != target.shape[0]:
            raise RuntimeError(f"[ERROR:NN] Target and Input should have same batch dimension. "
                               f"Shape of input: {x.shape}, target: {target.shape}")

        used_rows = 0
        batch_size = target.shape[0]

        # output = x.new_zeros(batch_size)  # target log probabilities (N,)
        output = torch.zeros(batch_size, dtype=torch.float32, device=x.device)
        gather_idx = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = torch.logical_and(torch.ge(target, low_idx), torch.lt(target, high_idx))
            row_indices = torch.nonzero(target_mask, as_tuple=True)[0]

            # no target is in batch
            if row_indices.numel() == 0:
                continue

            if i == 0:  # shortlist
                # keep as-is in head label
                gather_idx.index_copy_(0, row_indices, target[target_mask])
            else:
                relative_target = target[target_mask] - low_idx
                x_subset = x.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](x_subset)
                cluster_index = self.shortlist_size + i - 1

                # fill head label of cluster to cluster index
                gather_idx.index_fill_(0, row_indices, cluster_index)

                cluster_log_prob = F.log_softmax(cluster_output, dim=1)
                local_log_prob = cluster_log_prob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_log_prob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError(f"[ERROR:NN] Not all consumed. Should be {batch_size} but only used {used_rows}.")

        short_output = self.shortlist(x)  # (batch_size, shortlist_size)
        cluster_output = self.cluster(x)  # (batch_size, n_clusters)
        head_output = torch.cat([short_output, cluster_output], dim=1)  # (batch_size, shortlist_size + n_clusters)

        head_log_prob = F.log_softmax(head_output, dim=-1)
        # for word in shortlist: log(shortlist_prob_of_head)
        # for word in cluster: log(cluster_prob_of_head) + log(prob_in_cluster)
        output = output + head_log_prob.gather(1, gather_idx.unsqueeze(1)).squeeze()

        if self.reduction == "mean":
            loss = (-output).mean()
        elif self.reduction == "sum":
            loss = (-output).sum()
        else:
            loss = -output  # no reduction

        return output, loss

    def _get_full_log_prob(self, x: torch.Tensor, head_output: torch.Tensor) -> torch.Tensor:
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        batch_size = x.shape[0]

        out = x.new_empty(batch_size, self.num_classes)
        head_log_prob = F.log_softmax(head_output, dim=1)

        # copy shortlist prob
        out[:, :self.shortlist_size] = head_log_prob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](x)
            cluster_log_prob = F.log_softmax(cluster_output, dim=1)
            output_log_prob = cluster_log_prob + head_log_prob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_log_prob
        return out

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes log probabilities for all num_classes"""

        short_output = self.shortlist(x)  # (batch_size, shortlist_size)
        cluster_output = self.cluster(x)  # (batch_size, n_clusters)
        head_output = torch.cat([short_output, cluster_output], dim=1)  # (batch_size, shortlist_size + n_clusters)

        return self._get_full_log_prob(x, head_output)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ This is equivalent to `self.log_prob(input).argmax(dim=1)`"""

        short_output = self.shortlist(x)  # (batch_size, shortlist_size)
        cluster_output = self.cluster(x)  # (batch_size, n_clusters)
        head_output = torch.cat([short_output, cluster_output], dim=1)  # (batch_size, shortlist_size + n_clusters)

        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = torch.ge(output, self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(x, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(x[not_in_shortlist], head_output[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output

    def extra_repr(self) -> str:
        s = f"{self.num_classes}, {self.num_features}, cutoffs={self.cutoffs[:-1]}, div_value={self.div_value}"
        if self.use_bias:
            s += f", bias={self.use_bias}"
        if self.use_shortlist_proj:
            s += f", shortlist_proj={self.use_shortlist_proj}"
        return s


class AdaptiveEmbedding(BaseModule):

    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 cutoffs: Sequence[int],
                 div_value: float = 4.0,
                 padding_idx: Optional[int] = None,
                 shortlist_proj: bool = False,
                 word_drop_prob: float = 0.0):
        super(AdaptiveEmbedding, self).__init__()

        cutoffs = _preprocess_cutoffs(cutoffs, num_classes)

        self.num_features = num_features
        self.num_classes = num_classes

        self.cutoffs = cutoffs + [num_classes]  # [1000, 5000,] + [10000]: last index of each
        assert len(self.cutoffs) >= 2
        self.div_value = div_value
        self.use_shortlist_proj = shortlist_proj
        self.padding_idx = padding_idx

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1

        if not shortlist_proj:
            self.shortlist = Sequential(
                Embedding(self.shortlist_size, num_features, padding_idx=padding_idx,
                          word_drop_prob=word_drop_prob),
            )  # for consistency (nn.Sequential)
        else:
            self.shortlist = Sequential(
                Embedding(self.shortlist_size, num_features, padding_idx=padding_idx,
                          word_drop_prob=word_drop_prob),
                Linear(num_features, num_features, bias=False)
            )

        self.tail = ModuleList()
        for i in range(self.n_clusters):
            out_size = self.cutoffs[i + 1] - self.cutoffs[i]
            if div_value == 1:
                self.tail.append(Sequential(
                    Embedding(out_size, num_features, padding_idx=padding_idx,
                              word_drop_prob=word_drop_prob),
                ))  # for consistency (nn.Sequential)
            else:
                head_size = int(num_features / (self.div_value ** (i + 1)))
                self.tail.append(Sequential(
                    Embedding(out_size, head_size, padding_idx=padding_idx,
                              word_drop_prob=word_drop_prob),
                    Linear(head_size, num_features, bias=False),
                ))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        sequence_len, batch_size = indices.shape
        target = indices.view(-1)

        dtype = self.tail[0][-1].weight.dtype
        device = self.tail[0][-1].weight.device
        output = torch.zeros(sequence_len * batch_size, self.num_features, dtype=dtype, device=device)

        used_rows = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = torch.logical_and(torch.ge(target, low_idx), torch.lt(target, high_idx))
            row_indices = torch.nonzero(target_mask, as_tuple=True)[0]

            # no target is in batch
            if row_indices.numel() == 0:
                continue

            if i == 0:  # shortlist
                # keep as-is in head label
                proj = self.shortlist(target[target_mask])
                if output.dtype != proj.dtype:  # ad-hoc fix
                    output = output.to(proj.dtype)
                if output.device != proj.device:  # ad-hoc fix
                    output = output.to(proj.device)
                output.index_copy_(0, row_indices, proj)
            else:
                relative_target = target[target_mask] - low_idx
                proj = self.tail[i - 1](relative_target)
                output.index_copy_(0, row_indices, proj)

            used_rows += row_indices.numel()

        if used_rows != output.shape[0]:
            raise RuntimeError(f"[ERROR:NN] Not all consumed. Should be {output.shape[0]} but only used {used_rows}.")

        output = output.view(sequence_len, batch_size, self.num_features)
        # output *= float(math.sqrt(self.num_features))  # skip and externally handle.

        return output

    def extra_repr(self) -> str:
        s = f"{self.num_classes}, {self.num_features}, cutoffs={self.cutoffs[:-1]}, div_value={self.div_value}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.use_shortlist_proj:
            s += f", shortlist_proj=True"
        return s
