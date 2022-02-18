from datetime import datetime
import pprint

from .dist_utils import is_master


def time_log() -> str:
    a = datetime.now()
    return f"-" * 72 + f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}"


def print_log(*args, force_print: bool = False, **kwargs) -> None:
    """Print in case of (1) master (2) force_print=True"""
    if is_master() or force_print:
        if isinstance(args[-1], str) and args[-1].endswith("\n"):
            args = args[:-1] + (args[-1][:-1],)  # remove \n
        print(*args, **kwargs)


def print_network(net) -> None:
    s = "-" * 72 + "\n"
    s += net.__repr__()
    s += "-" * 72
    print_log(s)


def print_config(config):
    if is_master():
        print_log("-" * 72 + "\nConfig:")
        if is_master():
            try:
                pprint.pprint(config, sort_dicts=False)  # keep dict order. only supported Python >= 3.8
            except TypeError:
                pprint.pprint(config)
        print_log("-" * 72)
