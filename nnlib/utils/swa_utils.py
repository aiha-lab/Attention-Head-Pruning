import copy
import torch
import torch.nn as tnn


class SWAModel(object):

    def __init__(self, model: tnn.Module, device: str = "cpu"):
        super(SWAModel, self).__init__()
        self.model = copy.deepcopy(model).to(device)  # default to keep at CPU
        self.num = 0

    @torch.no_grad()
    def update_state(self, new_model: tnn.Module) -> None:
        # parameters to average, buffer to copy

        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            device = p_swa.device
            p_new_ = p_new.detach().to(device)
            if self.num == 0:
                p_swa.copy_(p_new_)
            else:
                # p' = (p_swa * n + p_new * 1) / (n + 1)
                # p' = p_swa + (p_new - p_swa) / (n + 1)
                p_avg = p_swa + (p_new_ - p_swa) / (self.num + 1)
                p_swa.copy_(p_avg)

        for b_swa, b_new in zip(self.model.buffers(), new_model.buffers()):
            device = b_swa.device
            b_new_ = b_new.detach().to(device)
            b_swa.copy_(b_new_)

        self.num += 1

    def state_dict(self) -> dict:
        state = dict()
        state["num_averaged"] = self.num
        state["network"] = self.model.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict["network"], strict=True)
        self.num = state_dict.get("num_averaged", 0)
