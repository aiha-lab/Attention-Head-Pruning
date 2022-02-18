from typing import Optional, Any
import torch
import torch.nn as tnn

from nnlib.utils.print_utils import print_log
from nnlib.common.collection import Collection


class _BaseModuleMixin(object):

    def __init__(self):
        self._name: Optional[str] = None
        self._collection: Optional[Collection] = None

    @property
    def name(self) -> Optional[str]:
        return self._name

    def set_name(self) -> None:
        for module_name, module in self.named_modules():
            if isinstance(module, BaseModule):
                module._name = module_name
        for param_name, param in self.named_parameters():
            param._name = param_name

    def set_collection(self, collection: Optional[Collection] = None) -> Collection:
        if collection is None:
            collection = Collection()  # instantiate new Collection
        for module in self.modules():
            if isinstance(module, BaseModule):
                module._collection = collection
        return collection

    def set_inplace(self, flag: bool = True) -> None:
        for module in self.modules():
            if hasattr(module, "inplace"):
                module.inplace = flag

    @staticmethod
    def print(*args, force_print: bool = False, **kwargs):
        print_log(*args, force_print=force_print, **kwargs)

    def collect(self, key: str, value: Optional[torch.Tensor]) -> None:
        if self._collection is None:
            self.print(f"[WARN:NN] collect({key}) is called, but collection is not set.")
            return
        self._collection[key] = value

    @property
    def collection(self):
        return self._collection


class BaseModule(tnn.Module, _BaseModuleMixin):
    def __init__(self):
        tnn.Module.__init__(self)
        _BaseModuleMixin.__init__(self)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    