from typing import Dict, Optional
import os
import platform
import torch.distributed as dist
import wandb

from nnlib.utils.dist_utils import is_master, broadcast_objects

__all__ = ["wandb_setup"]


def wandb_setup(cfg: Dict) -> Optional[str]:
    run_type = cfg["run_type"]
    save_dir = cfg["save_dir"]  # root save dir

    wandb_mode = cfg["wandb"]["mode"].lower()
    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"[ERROR:WANDB] Mode {wandb_mode} invalid.")

    if is_master():  # wandb init only at master
        os.makedirs(save_dir, exist_ok=True)

        wandb_project = cfg["project"]
        wandb_name = cfg["name"]

        wandb_note = cfg["wandb"]["notes"] if "notes" in cfg["wandb"] else None
        wandb_id = cfg["wandb"]["id"] if "id" in cfg["wandb"] else None
        server_name = platform.node()
        wandb_note = server_name + (f"-{wandb_note}" if (wandb_note is not None) else "")

        wandb.init(project=wandb_project, job_type=run_type, name=wandb_name, dir=save_dir,
                   resume="allow", mode=wandb_mode, notes=wandb_note, config=cfg, id=wandb_id)

        log_path = wandb.run.dir if (wandb_mode != "disabled") else save_dir
    else:
        log_path = None

    if dist.is_available() and dist.is_initialized():
        log_path = broadcast_objects([log_path], src_rank=0)[0]

    # if log_path is None:
    #     raise ValueError(f"[ERROR:DIST] log_path None, for {dist.get_rank()} GPU")
    return log_path
