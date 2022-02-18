from typing import Optional, Union
from numbers import Number
from collections import OrderedDict
import torch


class Collection(object):
    """Collection of key(str) and value(torch.Tensor / scalar).
    Does not support inter-GPU communication."""

    def __init__(self):
        self._col = OrderedDict()

    def __getitem__(self, key: str) -> Optional[Union[torch.Tensor, Number]]:
        try:
            return self._col[key]
        except KeyError:
            raise KeyError(f"[ERROR:COLLECTION] Key {key} is not in collection.")

    def __setitem__(self, key: str, value: Optional[Union[torch.Tensor, Number]]) -> None:
        # override possible
        if (value is not None) and (not isinstance(value, (torch.Tensor, Number))):
            raise ValueError(f"[ERROR:COLLECTION] Value of key {key} is not tensor nor number, got {type(value)}.")
        self._col[key] = value

    def keys(self):
        return self._col.keys()

    def values(self):
        return self._col.values()

    def items(self):
        return self._col.items()

    def clear(self) -> None:
        self._col.clear()
