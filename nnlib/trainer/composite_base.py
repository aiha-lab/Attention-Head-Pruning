from typing import Optional, List

from nnlib.nn.modules.module import BaseModule


class BaseCompositeModel(BaseModule):

    def __init__(self,
                 network: BaseModule,
                 losses: Optional[List[BaseModule]],
                 loss_coefficients: Optional[List[float]],
                 metrics: Optional[List[BaseModule]]):
        super(BaseCompositeModel, self).__init__()
        if network is None:
            raise ValueError("[ERROR:COMP] CompositeMode got None as network.")
        self.network = network

        if (losses is not None) and (loss_coefficients is not None):
            if len(losses) != len(loss_coefficients):
                raise ValueError("[ERROR:COMP] Loss and coefficients length mismatch.")
        elif (losses is not None) and (loss_coefficients is None):
            loss_coefficients = [1.0] * len(losses)
        else:  # losses is None, so coefficient is invalid.
            losses = loss_coefficients = None

        self.losses = losses
        self.loss_coefficients = loss_coefficients
        self.metrics = metrics

        self.network.set_name()
        col = self.set_collection()  # will create initial collection
        self.network.set_collection(collection=col)

    def __call__(self, *args, **kwargs) -> dict:
        self.collection.clear()
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        # we don"t return state_dict of losses nor metrics
        return self.network.state_dict(destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        # we don"t load state_dict of losses nor metrics
        return self.network.load_state_dict(state_dict, strict=strict)

    def register_forward_hook(self, hook):
        raise RuntimeError("[ERROR:COMP] hook is not supported for CompositeModel.")

    def register_forward_pre_hook(self, hook):
        raise RuntimeError("[ERROR:COMP] hook is not supported for CompositeModel.")

    def register_backward_hook(self, hook):
        raise RuntimeError("[ERROR:COMP] hook is not supported for CompositeModel.")
