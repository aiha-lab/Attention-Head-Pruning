from collections import OrderedDict
from torch.nn.parallel.distributed import DistributedDataParallel

__all__ = ["HappyDistributedDataParallel"]


class HappyDistributedDataParallel(DistributedDataParallel):
    """Simple wrapper of pytorch default DDP."""

    def state_dict(self, destination=None, prefix: str = '', keep_vars=False):
        return self.module.state_dict(destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def named_modules(self, memo=None, prefix: str = ''):
        return self.module.named_modules(memo=memo, prefix=prefix)

    def named_children(self):
        return self.module.named_children()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
