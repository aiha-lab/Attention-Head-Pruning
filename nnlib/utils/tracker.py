from typing import Union, List, Dict
import time
import torch

__all__ = ["TimeTracker", "TimeTrackerDict", "MetricTracker", "MetricTrackerDict"]

_time_func = time.perf_counter


class TimeTracker(object):

    def __init__(self) -> None:
        self.t = _time_func()
        self.duration = 0.0

    def reset(self) -> None:
        self.t = _time_func()
        self.duration = 0.0

    def update(self) -> float:
        new_t = _time_func()
        self.duration = new_t - self.t
        self.t = new_t
        return float(self.duration)

    def get(self) -> float:
        return float(self.duration)


class TimeTrackerDict(object):

    def __init__(self, *keys) -> None:
        self.time_dict = dict()
        for key in keys:
            self.time_dict[key] = TimeTracker()

    def _check_key(self, key: str):
        if key not in self.time_dict:
            raise KeyError(f"[ERROR:TRACK] TimeTracker key {key} is not in keys.")

    def reset(self, key=None) -> None:
        if key is None:  # reset all
            for key in self.time_dict.keys():
                self.time_dict[key].reset()
        else:
            self._check_key(key)
            self.time_dict[key].reset()

    def keys(self) -> List[str]:
        return list(self.time_dict.keys())

    def add_key(self, key: str) -> None:
        if key not in self.time_dict:
            self.time_dict[key] = TimeTracker()

    def add_keys(self, keys: List[str]) -> None:
        for k in keys:
            self.add_key(k)

    def update(self, key) -> float:
        self._check_key(key)
        return self.time_dict[key].update()

    def get(self, key) -> float:
        self._check_key(key)
        return self.time_dict[key].get()


class MetricTracker(object):

    def __init__(self) -> None:
        self.value: float = 0.0
        self.num: int = 0

    def reset(self) -> None:
        self.value = 0.0
        self.num = 0

    def update_add(self, value, num: int = 1) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value += value
        self.num += num

    def average(self) -> float:
        return float(self.value / self.num) if (self.num > 0) else 0

    def state_dict(self) -> dict:
        return {
            "value": self.value,  # float
            "num": self.num,  # int
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.value = state_dict.get("value", 0.0)
        self.num = state_dict.get("num", 0)


class MetricTrackerDict(object):

    def __init__(self, *keys) -> None:
        self.value_dict = dict()
        for key in keys:
            self.value_dict[key] = MetricTracker()

    def _check_key(self, key: str):
        if key not in self.value_dict:
            raise KeyError(f"[ERROR:TRACK] MetricTracker key {key} is not in keys.")

    def reset(self) -> None:
        for key in self.value_dict.keys():
            self.value_dict[key].reset()

    def update_add(self, value_dict: dict, default_num: int = 1) -> None:
        for key, value in value_dict.items():
            self._check_key(key)
            if isinstance(value, tuple):
                assert len(value) == 2
                self.value_dict[key].update_add(value[0], num=value[1])
            else:
                self.value_dict[key].update_add(value, num=default_num)

    def keys(self) -> List[str]:
        return list(self.value_dict.keys())

    def add_key(self, key: str) -> None:
        if key not in self.value_dict:
            self.value_dict[key] = MetricTracker()

    def add_keys(self, keys: List[str]) -> None:
        for k in keys:
            self.add_key(k)

    def avg(self, key=None) -> Union[dict, float]:
        if key is None:
            avg = dict()
            for key in self.value_dict.keys():
                avg[key] = self.value_dict[key].average()
            return avg
        else:
            self._check_key(key)
            return self.value_dict[key].average()

    def average(self, key=None):
        return self.avg(key)  # backward compatibility

    def value(self, key=None) -> Union[dict, float]:
        if key is None:
            val = dict()
            for key in self.value_dict.keys():
                val[key] = self.value_dict[key].value
            return val
        else:
            self._check_key(key)
            return self.value_dict[key].value

    def num(self, key=None) -> Union[dict, int]:
        if key is None:
            n = dict()
            for key in self.value_dict.keys():
                n[key] = self.value_dict[key].num
            return n
        else:
            self._check_key(key)
            return self.value_dict[key].num

    def state_dict(self) -> Dict:
        states = {}
        for key in self.value_dict.keys():
            states[key] = self.value_dict[key].state_dict()
        return states

    def load_state_dict(self, state_dict: Dict) -> None:
        # only load key that matches current dict. others will be discarded.
        for k in state_dict.keys():
            if k in self.value_dict.keys():
                self.value_dict[k].load_state_dict(state_dict[k])
