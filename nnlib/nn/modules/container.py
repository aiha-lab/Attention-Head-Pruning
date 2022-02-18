from typing import Any, Union, Iterator, Optional, Iterable, Dict, Tuple
import operator
from itertools import islice
from collections import OrderedDict
import torch
import torch.nn as nn

from nnlib.nn.modules.module import BaseModule


class Sequential(BaseModule):

    def __init__(self, *args: Any) -> None:
        super(Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx: int):
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"[ERROR:NN] Sequential index {idx} is out of range")
        idx = idx % size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            x = module(x)
        return x


class ModuleList(BaseModule):

    def __init__(self, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super(ModuleList, self).__init__()
        if modules is not None:
            self.extend(modules)

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"[ERROR:NN] index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: int) -> nn.Module:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[nn.Module]):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: nn.Module) -> None:
        """Insert a given module before a given index in the list."""
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: nn.Module):
        """Appends a given module to the end of the list."""
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[nn.Module]):
        """Appends modules from a Python iterable to the end of the list."""
        if not isinstance(modules, Iterable):
            raise TypeError("[ERROR:NN] ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def forward(self):
        raise NotImplementedError()


class ModuleDict(BaseModule):

    def __init__(self, modules: Optional[Dict[str, nn.Module]] = None) -> None:
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> nn.Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: nn.Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> nn.Module:
        """Remove key from the ModuleDict and return its module."""
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        """Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    def items(self) -> Iterable[Tuple[str, nn.Module]]:
        """Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    def values(self) -> Iterable[nn.Module]:
        """Return an iterable of the ModuleDict values."""
        return self._modules.values()

    def update(self, modules: Dict[str, nn.Module]) -> None:
        """Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys."""
        if not isinstance(modules, Iterable):
            raise TypeError("[ERROR:NN] ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, Dict)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules.items()):
                if not isinstance(m, Iterable):
                    raise TypeError("[ERROR:NN] ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("[ERROR:NN] ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                m_key, m_value = m
                self[m_key] = m_value

    def forward(self):
        raise NotImplementedError
