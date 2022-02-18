from typing import Tuple, Any, Optional, Union, Dict


def to_tuple(x: Any) -> Optional[Union[Dict, Tuple]]:
    if x is None:
        return None
    if isinstance(x, (dict, tuple)):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def to_n_tuple(x, n: int):
    if isinstance(x, (tuple, list)):
        # if len(x) != n:
        #     raise ValueError(f'x should be length {n} but got {len(x)}.')
        return tuple(x)
    return (x,) * n


def to_single_tuple(x):
    return to_n_tuple(x, 1)


def to_pair_tuple(x):
    return to_n_tuple(x, 2)


def to_triple_tuple(x):
    return to_n_tuple(x, 3)


def to_quadruple_tuple(x):
    return to_n_tuple(x, 4)


