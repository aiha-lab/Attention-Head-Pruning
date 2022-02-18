import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seed(seed: int, benchmark: bool = True, deterministic: bool = False) -> None:
    random.seed(seed + 1)
    np.random.seed(seed + 2)
    torch.manual_seed(seed + 3)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    # os.environ["PYTHONHASHSEED"] = str(seed + 6)
    if benchmark:
        cudnn.benchmark = True
    if deterministic:
        print("[WARN:SEED] CUDNN Deterministic is set to True, which may cause significant slowdown.")
        cudnn.deterministic = True
