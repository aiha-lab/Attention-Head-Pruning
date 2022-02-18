from typing import Any, Dict


class BaseTransform(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: Any) -> Any:
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        raise NotImplementedError
