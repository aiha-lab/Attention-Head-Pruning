from typing import Dict, Any, Optional, List

from nnlib.utils.dist_utils import get_world_size, get_rank


def override_config(base: Dict, override: Dict) -> Dict:
    for key in override:
        # new key
        if key not in base:
            base[key] = override[key]  # add
            continue
        # existing key
        if key in base:
            if isinstance(base[key], Dict) and isinstance(override[key], Dict):
                if ("name" in base[key]) and ("name" in override[key]):  # both ar builder-based
                    if base[key]["name"] == override[key]["name"]:  # same name, only some will be changed
                        override_config(base[key], override[key])
                    else:
                        base[key] = override[key]  # replace
                elif ("name" not in base[key]) and ("name" not in override[key]):  # both are not builder-based
                    override_config(base[key], override[key])
                elif ("name" in base[key]) and ("name" not in override[key]):  # assume same name
                    override_config(base[key], override[key])
                else:
                    raise ValueError(f"[ERROR:CONFIG] Override {key} named config with non-named one.")
            elif (not isinstance(base[key], Dict)) and (not isinstance(override[key], Dict)):
                if isinstance(base[key], List) and isinstance(override[key], List):
                    for b, o in zip(base[key], override[key]):
                        if isinstance(b, Dict) and isinstance(o, Dict):
                            override_config(b, o)
                else:
                    base[key] = override[key]  # replace
            else:
                raise ValueError(f"[ERROR:CONFIG] Override {key} dict config with non-dict one.")

    return base  # Dict is reference, so actually we don't have to return.


def override_config_by_key_value(base: Dict, key: str, value: Any,
                                 exclude_str: Optional[str] = None) -> Dict:
    """Find key in any place in config and replace"""
    for k in base:
        if (exclude_str is not None) and (exclude_str in k):
            continue
        if k == key:
            base[k] = value
        elif isinstance(base[k], Dict):
            override_config_by_key_value(base[k], key, value)
        elif isinstance(base[k], List):
            for b in base[k]:
                if isinstance(b, Dict):
                    override_config_by_key_value(b, key, value)
    return base


def override_config_by_parse(base: Dict, script_args: List[str]):
    """Usage of script_args:
    A.B=C D=E F.G.H=K
    [A.B=C, D=E, F.G.H=K]
    Maximum 4-depth is supported & cannot override list-type values.
    Value should not contain '.' and ' ' (space).
    """

    def _require_eval(x):
        return isinstance(x, (int, float, bool))

    for a in script_args:
        a: str
        kv = a.split("=")
        if len(kv) != 2:
            raise ValueError(f"[ERROR:CONFIG] CLI override should formulated as X.Y=Z, but got {a}.")
        key = kv[0]
        value = kv[1]
        if value[0] in ("(", "[", "{"):
            value = eval(value)

        ck = key.split(".")
        if len(ck) > 4:
            raise ValueError(f"[ERROR:CONFIG] CLI override is currently supported for maximum depth 4, "
                             f"but got {a} (depth: {len(ck)}).")
        try:
            if value == "true":
                value = "True"
            elif value == "false":
                value = "False"
            # value is str, we should convert it to match actual type.

            if len(ck) == 1:
                if _require_eval(base[ck[0]]):
                    value = eval(value)
                base[ck[0]] = value
            elif len(ck) == 2:
                if _require_eval(base[ck[0]][ck[1]]):
                    value = eval(value)
                base[ck[0]][ck[1]] = value
            elif len(ck) == 3:
                if _require_eval(base[ck[0]][ck[1]][ck[2]]):
                    value = eval(value)
                base[ck[0]][ck[1]][ck[2]] = value
            elif len(ck) == 4:
                if _require_eval(base[ck[0]][ck[1]][ck[2]][ck[3]]):
                    value = eval(value)
                base[ck[0]][ck[1]][ck[2]][ck[3]] = value
        except KeyError:
            raise KeyError(f"[ERROR:CONFIG] Key {ck} is not in config.")


def split_data_config(config: Dict,
                      world_size: Optional[int] = None,
                      local_rank: Optional[int] = None) -> Dict:
    """Split config elements that are related to number of samples/tokens in each GPUs.
    We decide to use (1) full data load for each GPU (2) use DistributedSampler
    """
    if world_size is None:
        world_size = get_world_size()
    if local_rank is None:
        local_rank = get_rank()

    # split batch_size and num_workers in config items
    for key in config:
        if isinstance(config[key], Dict):
            split_data_config(config[key], world_size, local_rank)
        elif "batch_size" in key:
            batch_size = config[key]
            config[key] = int(max(batch_size // world_size, 1))
        elif "num_workers" in key:
            num_workers = config[key]
            if num_workers > 0:
                config[key] = int(max((num_workers + world_size - 1) // world_size, 1))
        elif "max_tokens" in key:
            max_tokens = config[key]
            if max_tokens > 0:
                config[key] = int(max(max_tokens // world_size, 1))
        elif "max_sentences" in key:
            max_sentences = config[key]
            if max_sentences > 0:
                config[key] = int(max(max_sentences // world_size, 1))
        elif "max_token_by_feature" in key:
            max_token_by_feature = config[key]
            if max_token_by_feature > 0:
                config[key] = int(max(max_token_by_feature // world_size, 1))
        elif "max_samples" in key:
            max_samples = config[key]
            if max_samples > 0:
                config[key] = int(max(max_samples // world_size, 1))
    return config
