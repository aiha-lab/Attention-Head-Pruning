import torch

from all_transformer_xl import AllTransformerLM
from all_transformerl_xl_infer import AllTransformerLMInfer
from nnlib.utils.tracker import TimeTracker

from nnlib.models.transformer_xl import MemoryTransformer

hidden_dim = 512
num_heads = 8
feedforward_dim = 2048

num_tokens = 267735
div_value = 4
cutoffs = (20000, 40000, 200000)
num_layers = 16
seq_len = 192

# num_tokens = 28
# div_value = 1
# cutoffs = ()
# num_layers = 12
# seq_len = 512

pre_norm = True

# ======================================================================================================== #
# ======================================================================================================== #

base_network = AllTransformerLM(num_tokens, num_layers, hidden_dim, num_heads, feedforward_dim,
                                target_len=seq_len, memory_len=seq_len, extend_len=0,
                                attn_bias=False, share_r_bias=False, div_value=div_value, tie_weight=True,
                                tie_proj=True,
                                cutoffs=cutoffs, pre_norm=pre_norm)
base_network.eval()
base_network.to('cuda')

effective_heads = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# effective_heads = [8, 3, 7, 6, 6, 6, 6, 7, 6, 7, 8, 8, 7, 8, 7, 6]  # 0.01
# effective_heads = [6, 4, 6, 5, 4, 5, 5, 6, 5, 6, 7, 5, 6, 6, 6, 5]  # 0.015
# effective_heads = [5, 2, 5, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3]  # 0.02

infer_network = AllTransformerLMInfer(num_tokens, num_layers, hidden_dim, num_heads, feedforward_dim,
                                      target_len=seq_len, memory_len=seq_len, extend_len=0,
                                      attn_bias=False, share_r_bias=False, effective_heads=effective_heads,
                                      div_value=div_value, tie_weight=True, tie_proj=True,
                                      cutoffs=cutoffs, pre_norm=pre_norm)
infer_network.eval()
infer_network.to('cuda')

# xl_dim = 384
#
# xl_network = MemoryTransformer(num_tokens, num_layers, xl_dim, num_heads, xl_dim * 4,
#                                seq_length=seq_len, mem_length=seq_len, div_value=1, cutoffs=cutoffs,
#                                pre_norm=False, same_length=True)
# xl_network.eval()
# xl_network.to('cuda')

# ======================================================================================================== #
# ======================================================================================================== #

batch_size = 4
steps = 100
data_length = 192 * steps

tracker = TimeTracker()
data = torch.randint(0, num_tokens, (batch_size, data_length + 1,), device='cuda').long()

duration = 0
with torch.no_grad():
    memory = None
    for step_idx in range(steps):
        if step_idx % 10 == 0:
            print('Step: {} / {}'.format(step_idx, steps))
        seq_input = data[:, step_idx * 192:(step_idx + 1) * 192].contiguous()
        seq_target = data[:, step_idx * 192 + 1:(step_idx + 1) * 192 + 1].contiguous()
        # seq_target = None

        tracker.reset()

        _, _, _, memory = base_network.forward(seq_input, seq_target, memory)
        torch.cuda.synchronize()

        duration += tracker.update()

print("Baseline network:", duration / steps)

# ======================================================================================================== #
# ======================================================================================================== #

duration = 0
with torch.no_grad():
    memory = None
    for step_idx in range(steps):
        if step_idx % 10 == 0:
            print('Step: {} / {}'.format(step_idx, steps))
        seq_input = data[:, step_idx * 192:(step_idx + 1) * 192].contiguous()
        seq_target = data[:, step_idx * 192 + 1:(step_idx + 1) * 192 + 1].contiguous()
        # seq_target = None

        tracker.reset()

        _, memory = infer_network.forward(seq_input, seq_target, memory)
        torch.cuda.synchronize()

        duration += tracker.update()

print("Infer network:", duration / steps)

# ======================================================================================================== #
# ======================================================================================================== #
#
# duration = 0
# with torch.no_grad():
#     memory = None
#     for step_idx in range(steps):
#         if step_idx % 10 == 0:
#             print('Step: {} / {}'.format(step_idx, steps))
#         seq_input = data[:, step_idx * 192:(step_idx + 1) * 192].contiguous()
#         seq_target = data[:, step_idx * 192 + 1:(step_idx + 1) * 192 + 1].contiguous()
#         # seq_target = None
#
#         tracker.reset()
#
#         _, memory = xl_network(seq_input, seq_target, memory)
#         torch.cuda.synchronize()
#
#         duration += tracker.update()
#
# print("XL network:", duration / steps)
