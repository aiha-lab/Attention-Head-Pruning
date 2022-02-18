from typing import Optional, Union, List, Tuple, Dict, Any, Iterable
import os
import copy
import contextlib
import torch
import torch.distributed as dist
import torch.cuda.amp as amp

from nnlib.nn.modules import BaseModule
from nnlib.nn.parallel.distributed import HappyDistributedDataParallel
from nnlib.data.dataloaders import HappyDataLoader
from nnlib.optim.optimizer import BaseOptimizer, OptimizerList
from nnlib.optim.lr_scheduler.scheduler import BaseScheduler, SchedulerList

from nnlib.utils.dist_utils import init_distributed, get_rank, get_world_size
from nnlib.utils.seed_utils import init_seed
from nnlib.utils.print_utils import print_log, time_log
from nnlib.utils.param_utils import compute_param_norm, count_params
from nnlib.utils.grad_utils import compute_grad_norm, DummyGradScaler
from nnlib.utils.swa_utils import SWAModel
from nnlib.utils.tracker import TimeTrackerDict, MetricTrackerDict


class BaseTrainer(object):

    def __init__(self,
                 model: BaseModule,
                 save_dir: str,
                 train_dataloader: Optional[HappyDataLoader] = None,
                 valid_dataloader: Optional[HappyDataLoader] = None,
                 test_dataloader: Optional[HappyDataLoader] = None,
                 optimizer: Optional[Union[BaseOptimizer, OptimizerList]] = None,
                 scheduler: Optional[Union[BaseScheduler, SchedulerList]] = None,
                 gpus: int = 1,
                 seed: int = 1234,
                 fp16: bool = False,
                 max_epochs: int = 1000,
                 max_iterations: int = 10000000,
                 print_interval_iters: int = 50,
                 log_interval_iters: Optional[int] = None,
                 valid_interval_epochs: Union[int, float] = 1,
                 clip_grad_norm: float = 0.0,
                 accumulate_num_batches: int = 1,
                 accumulate_num_samples: int = -1,
                 start_epoch: int = 0,
                 start_iteration: int = 0,
                 *,
                 cudnn_benchmark: bool = True,
                 cudnn_deterministic: bool = False,
                 fp16_init_scale: float = 4096.0,
                 fp16_growth_interval: Optional[int] = 1000,
                 run_test_with_valid: bool = False,
                 final_test_mode: str = "best",
                 find_unused_parameters: bool = False,
                 ckpt_save_latest: bool = True,
                 ckpt_save_best: bool = True,
                 ckpt_swa_start_epoch: int = -1,
                 ckpt_swa_start_iters: int = -1,
                 verbose: bool = True) -> None:
        # -------------------------------------------------------------------------------------- #
        self.current_epoch = start_epoch
        self.current_iteration = start_iteration  # global iteration

        self.save_dir = save_dir
        self.verbose = verbose

        self.max_epochs = max_epochs
        self.max_iterations = max_iterations
        self.print_interval_iters = print_interval_iters
        self.log_interval_iters = log_interval_iters if (log_interval_iters is not None) else print_interval_iters
        self.valid_interval_epochs = valid_interval_epochs  # if int, epoch / float: ratio
        self.clip_grad_norm = max(clip_grad_norm, 0.0)
        self.accumulate_num_batches = max(accumulate_num_batches, 1)

        if accumulate_num_samples > 0:  # higher priority
            self.accumulate_num_batches = -1
        self.accumulate_num_samples = accumulate_num_samples
        self._accumulated_samples = 0
        self._accumulated_batches = 0

        # -------------------------------------------------------------------------------------- #
        # distributed setup
        self.device, self.local_rank, self.world_size, self.is_distributed = self.distributed_setup(gpus)
        self.is_master = (self.local_rank == 0)
        model = model.to(self.device)
        if self.is_distributed and not isinstance(model, HappyDistributedDataParallel):
            model = HappyDistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank,
                                                 find_unused_parameters=find_unused_parameters)
        self.model = model

        # -------------------------------------------------------------------------------------- #
        # seed setup
        init_seed(seed + self.local_rank, benchmark=cudnn_benchmark, deterministic=cudnn_deterministic)

        # -------------------------------------------------------------------------------------- #
        # dataloader, optimizer, scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader if (valid_dataloader is not None) else test_dataloader
        self.test_dataloader = test_dataloader if (test_dataloader is not None) else valid_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        if hasattr(self.scheduler, "verbose"):  # override
            self.scheduler.verbose = verbose
        if hasattr(self.scheduler, "num_iterations"):
            self.scheduler.num_iterations = start_iteration - 1

        # -------------------------------------------------------------------------------------- #
        # ckpt setup
        self.ckpt_save_latest = ckpt_save_latest
        self.ckpt_save_best = ckpt_save_best
        if save_dir is not None:
            self.ckpt_best_path = os.path.join(save_dir, "best.ckpt")
            self.ckpt_latest_path = os.path.join(save_dir, "latest.ckpt")
            self.ckpt_swa_path = os.path.join(save_dir, "swa.ckpt")
        else:
            self.ckpt_best_path = self.ckpt_latest_path = self.ckpt_swa_path = None

        # -------------------------------------------------------------------------------------- #
        # fp16 setup
        self.fp16 = fp16
        if fp16:
            self.grad_scaler = amp.GradScaler(init_scale=fp16_init_scale, growth_interval=fp16_growth_interval)
        else:
            self.grad_scaler = DummyGradScaler()

        # -------------------------------------------------------------------------------------- #
        # swa setup
        self.ckpt_swa_start_epoch = ckpt_swa_start_epoch
        self.ckpt_swa_start_iters = ckpt_swa_start_iters
        self.swa_model = SWAModel(self.model, device="cpu") if (
                ckpt_swa_start_epoch >= 0 or ckpt_swa_start_iters >= 0) else None

        # -------------------------------------------------------------------------------------- #
        # test and swa setup
        self.run_test_with_valid = run_test_with_valid
        self.final_test_mode = final_test_mode.lower()
        if self.final_test_mode not in ("latest", "best", "swa"):
            raise ValueError(f"[ERROR:TRAINER] Final test mode {self.final_test_mode} not supported.")
        if (self.final_test_mode == "best") and (self.ckpt_best_path is None):
            raise ValueError(f"[ERROR:TRAINER] Final test mode BEST set but ckpt_best_path is None.")
        if (self.final_test_mode == "swa") and ((self.swa_model is None) or (self.ckpt_swa_path is None)):
            raise ValueError(f"[ERROR:TRAINER] Final test mode SWA set but swa_model is None.")

        # -------------------------------------------------------------------------------------- #
        # tracker setup
        self.time_tracker = TimeTrackerDict("epoch", "iter", "valid", "test",
                                            "data", "forward", "backward", "optimizer")

        self.common_tracker = MetricTrackerDict("num_samples")  # tracker that is not reset through epoch
        self.train_tracker = MetricTrackerDict("loss")
        self.valid_tracker = MetricTrackerDict("loss")
        self.test_tracker = MetricTrackerDict("loss")

        # -------------------------------------------------------------------------------------- #
        # extra
        self.init_additional()
        self._state_loaded = False

    def init_additional(self):
        # can be override by child class
        pass

    def reset_save_dir(self, save_dir: str):
        self.save_dir = save_dir
        if save_dir is not None:
            self.ckpt_best_path = os.path.join(save_dir, "best.ckpt")
            self.ckpt_latest_path = os.path.join(save_dir, "latest.ckpt")
            self.ckpt_swa_path = os.path.join(save_dir, "swa.ckpt")
        else:
            self.ckpt_best_path = self.ckpt_latest_path = self.ckpt_swa_path = None

    def print(self, *args, force_print: bool = False, **kwargs):
        if self.verbose or force_print:
            print_log(*args, force_print=True, **kwargs)

    def add_num_samples(self, count: int) -> None:
        if self._accumulated_samples >= self.accumulate_num_samples:
            self._accumulated_samples = 0  # reset, should already be used.
            self._accumulated_batches = 0
        self._accumulated_samples += count
        self._accumulated_batches += 1

    def is_zero_grad_iter(self, num_iter: int) -> bool:
        # helper to use within train_epoch_body
        if self.accumulate_num_batches > 0:
            return num_iter % self.accumulate_num_batches == 0
        else:  # use sample
            y = (self._accumulated_samples <= 0)
            return y

    def is_step_iter(self, num_iter: int) -> bool:
        # helper to use within train_epoch_body
        if self.accumulate_num_batches > 0:
            return (num_iter % self.accumulate_num_batches) == (self.accumulate_num_batches - 1)
        else:  # use sample
            y = (self._accumulated_samples >= self.accumulate_num_samples)
            return y

    @property
    def accumulated_samples(self) -> int:
        return self._accumulated_samples

    @property
    def accumulated_batches(self) -> int:
        return self._accumulated_batches

    def is_print_iter(self, num_iter: int) -> bool:
        # helper to use within train_epoch_body, valid_body
        return (num_iter % self.print_interval_iters == 0) and (num_iter > 0)

    def is_log_iter(self, num_iter: int) -> bool:
        # helper to use within train_epoch_body
        return num_iter % self.log_interval_iters == 0

    def is_valid_iter(self, num_iter: int, num_dataloader: int) -> bool:
        # helper to use within train_epoch_body
        c = int(self.valid_interval_epochs * num_dataloader)
        return (0 < self.valid_interval_epochs < 1) and (num_iter > 0) and (num_iter % c == 0)

    def is_swa_update(self) -> bool:
        if self.swa_model is None:
            return False

        # True if either condition is satisfied.
        is_epoch = (self.current_epoch >= self.ckpt_swa_start_epoch) if (self.ckpt_swa_start_epoch >= 0) else False
        is_iter = (self.current_iteration >= self.ckpt_swa_start_iters) if (self.ckpt_swa_start_iters >= 0) else False
        return is_epoch or is_iter

    def maybe_fp16(self):
        # helper to use within train_epoch_body
        if self.fp16:
            return amp.autocast()
        else:
            try:
                return contextlib.nullcontext()  # available at Python >= 3.7
            except AttributeError:
                return contextlib.suppress()  # available at Python >= 3.4

    def maybe_no_sync(self, num_iter: int):
        # helper to use within train_epoch_body
        if isinstance(self.model, HappyDistributedDataParallel) and (not self.is_step_iter(num_iter)):
            return self.model.no_sync()
        else:
            try:
                return contextlib.nullcontext()  # available at Python >= 3.7
            except AttributeError:
                return contextlib.suppress()  # available at Python >= 3.4

    def compute_grad_norm(self) -> torch.Tensor:
        return compute_grad_norm(self.model.parameters(), self.clip_grad_norm)

    def compute_param_norm(self) -> torch.Tensor:
        return compute_param_norm(self.model.parameters())

    def check_grad(self) -> None:
        """Check Gradient status. WARN: THIS MAKES NETWORK VERY SLOW."""
        for p_name, p in self.model.named_parameters():
            if p.grad is None:
                self.print(f"[WARN:TRAINER] Parameter {p_name} grad is None.")
            elif torch.any(torch.isnan(p.grad)):
                self.print(f"[WARN:TRAINER] Parameter {p_name} grad is NaN.")
            elif torch.any(torch.isinf(p.grad)):
                self.print(f"[WARN:TRAINER] Parameter {p_name} grad is Inf.")

    def train(self,
              train_dataloader: Optional[Iterable] = None,
              valid_dataloader: Optional[Iterable] = None,
              test_dataloader: Optional[Iterable] = None):
        # ------------------------------------------------------------ #
        if train_dataloader is None:
            train_dataloader = self.train_dataloader
        if valid_dataloader is None:
            valid_dataloader = self.valid_dataloader
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if train_dataloader is None:
            self.print("[WARN:TRAINER] SKIP train run because train_dataloader is None.")
            return
        # ------------------------------------------------------------ #
        self.on_train_start()
        self.scheduler.step()

        if self._state_loaded:  # loaded from somewhere
            self.valid(valid_dataloader, test_dataloader)
        # ------------------------------------------------------------ #
        try:
            while (self.current_epoch < self.max_epochs) and (self.current_iteration < self.max_iterations):
                # ---------------------------------------------- #
                if self.is_distributed:
                    try:
                        train_dataloader.set_epoch(self.current_epoch)
                    except AttributeError:
                        self.print("[WARN:TRAINER] Train dataloader does not support set_epoch. Is this correct?")
                # ---------------------------------------------- #
                self.on_train_epoch_start()
                # ---------------------------------------------- #
                # don't forget to increase self.current_iteration inside
                self.train_epoch_body(train_dataloader)
                # ---------------------------------------------- #
                if isinstance(self.valid_interval_epochs, int) and (
                        self.current_epoch % self.valid_interval_epochs == 0):
                    # valid per n epoch(s)
                    self.valid(valid_dataloader, test_dataloader, track=True)
                elif 0 < self.valid_interval_epochs < 1:
                    # valid inside train_body & at the end of every epoch
                    self.valid(valid_dataloader, test_dataloader, track=True)
                # ---------------------------------------------- #
                self.on_train_epoch_end()
                # ---------------------------------------------- #
                self.current_epoch += 1
        except KeyboardInterrupt:
            self.print("[ERROR:TRAINER] KeyboardInterrupt detected during training.")
        # ------------------------------------------------------------ #
        self.on_train_end()
        # ------------------------------------------------------------ #
        self.test(test_dataloader, mode=self.final_test_mode)
        # ------------------------------------------------------------ #

    def valid(self,
              valid_dataloader: Optional[Iterable] = None,
              test_dataloader: Optional[Iterable] = None,
              *, track: bool = True, raise_oom: bool = False):
        # ------------------------------------------------------------ #
        if valid_dataloader is None:
            valid_dataloader = self.valid_dataloader
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if valid_dataloader is None:
            self.print("[WARN:TRAINER] SKIP valid run because valid_dataloader is None.")
            return
        # ------------------------------------------------------------ #
        self.on_valid_start()
        # ------------------------------------------------------------ #
        try:
            self.valid_body(valid_dataloader, track=track)
        except RuntimeError as e:  # adapted from FairSeq trainer/valid_step
            if "out of memory" in str(e).lower():
                if not raise_oom:
                    self.print(f"[WARN:TRAINER] OOM during valid, retrying with memory free.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad = None  # free some memory
                    torch.cuda.empty_cache()
                    return self.valid(valid_dataloader, test_dataloader, track=track, raise_oom=True)
            raise e
        # ------------------------------------------------------------ #
        self.on_valid_end()
        # ------------------------------------------------------------ #
        if self.run_test_with_valid:
            if test_dataloader is None:
                self.print("[WARN:TRAINER] Set to run test with valid, but test_dataloader is None.")
            self.test(test_dataloader)
        # ------------------------------------------------------------ #

    def test(self,
             test_dataloader: Optional[Iterable] = None,
             *, mode: str = "latest"):
        # ------------------------------------------------------------ #
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if test_dataloader is None:
            self.print("[WARN:TRAINER] SKIP test run because test_dataloader is None.")
            return
        # ------------------------------------------------------------ #
        mode = mode.lower()
        # WARNING: test with (mode != latest) loads parameter from past state, and do not recover current state.
        if not self._state_loaded:  # ignore mode when state is loaded
            if mode == "latest":
                self.print("[LOG:TRAINER] Test with 'latest' state_dict.")
            elif mode == "best":
                if self.ckpt_best_path is not None:
                    try:
                        self.load_state_dict(torch.load(self.ckpt_best_path, map_location=self.device),
                                             strict=True, model_only=False)
                        self.print("[LOG:TRAINER] Test with 'best' state_dict.")
                    except FileNotFoundError:  # best not yet created
                        self.print("[LOG:TRAINER] Test with 'best' state_dict requested but failed (no file), "
                                   "run with 'latest'.")
                else:
                    self.print("[LOG:TRAINER] Test with 'best' state_dict requested but failed (best not set), "
                               "run with 'latest'.")
            elif mode == "swa":
                if (self.ckpt_swa_path is not None) and (self.swa_model is not None):
                    try:
                        self.load_state_dict(torch.load(self.ckpt_swa_path, map_location=self.device),
                                             strict=True, model_only=True)
                        self.print("[LOG:TRAINER] Test with 'swa' state_dict.")
                    except FileNotFoundError:  # swa not yet created
                        self.print("[LOG:TRAINER] Test with 'swa' state_dict requested but failed (no file), "
                                   "run with 'latest'.")
                else:
                    self.print("[LOG:TRAINER] Test with 'swa' state_dict requested but failed (swa not set), "
                               "run with 'latest'.")
            else:
                self.print(f"[LOG:TRAINER] Test mode {mode} state_dict is invalid, run with 'latest'.")
        else:  # _state_loaded (from outside)
            self.print(f"[LOG:TRAINER] Test with loaded state_dict.")
        # ------------------------------------------------------------ #
        self.on_test_start()
        # ------------------------------------------------------------ #
        self.test_body(test_dataloader)
        # ------------------------------------------------------------ #
        self.on_test_end()
        # ------------------------------------------------------------ #

    def train_epoch_body(self, dataloader):
        # don"t forget to call model.train(), with maybe_fp16(), ...
        pass

    def valid_body(self, dataloader, *, track: bool = True):
        # don"t forget to call model.eval(), with torch.no_grad(), ...
        pass

    def test_body(self, dataloader):
        # don"t forget to call model.eval(), with torch.no_grad(), ...
        pass

    def on_train_start(self):
        s = time_log() + "\n"
        s += "Train start!\n"
        # print simple statistic of network
        param_num, param_count = count_params(self.model.parameters())
        param_norm = compute_param_norm(self.model.parameters())
        s += f"... Number of parameters: {param_num}, elements: {param_count}\n" \
             f"... Initial norm of parameters: {param_norm.item():.4f}"
        self.print(s)

    def on_train_end(self):
        s = time_log() + "\n"
        s += f"Train done! " \
             f"Final epoch {self.current_epoch} / {self.max_epochs}, " \
             f"total iterations {self.current_iteration} / {self.max_iterations}"
        self.print(s)

    def on_valid_start(self):
        s = time_log() + "\n"
        s += f"Valid at epoch {self.current_epoch} / {self.max_epochs}, iteration {self.current_iteration} start!"
        self.print(s)
        self.time_tracker.reset("valid")

    def on_valid_end(self):
        valid_time = self.time_tracker.update("valid")
        s = time_log() + "\n"
        s += f"Valid done! (Time: {valid_time:.4f} s)"
        self.print(s)

    def on_test_start(self):
        s = time_log() + "\n"
        s += "Test start!"
        self.print(s)
        self.time_tracker.reset("test")

    def on_test_end(self):
        test_time = self.time_tracker.update("test")
        s = time_log() + "\n"
        s += f"Test done! (Time: {test_time:.4f} s)"
        self.print(s)

    def on_train_epoch_start(self):
        s = time_log() + "\n"
        s += f"Train epoch {self.current_epoch} / {self.max_epochs} " \
             f"(iteration {self.current_iteration} / {self.max_iterations}) start!"
        self.print(s)
        self.time_tracker.reset()  # reset all

        self._accumulated_samples = 0
        self._accumulated_batches = 0

    def on_train_epoch_end(self):
        epoch_time = self.time_tracker.update("epoch")
        s = time_log() + "\n"
        s += f"Train epoch done! (Time: {epoch_time:.4f} s)"
        self.print(s)

    def state_dict(self) -> dict:
        state = dict()
        state["epoch"] = self.current_epoch
        state["iteration"] = self.current_iteration

        state["network"] = self.model.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scheduler"] = self.scheduler.state_dict()

        state["tracker"] = {
            "common": self.common_tracker.state_dict(),
            "train": self.train_tracker.state_dict(),
            "valid": self.valid_tracker.state_dict(),
            "test": self.test_tracker.state_dict(),
        }

        if self.fp16:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True, model_only: bool = False,
                        load_keys: Optional[List] = None, ignore_keys: Optional[List] = None) -> None:
        """Load state dict.
        Possible keys:
            [network, optimizer, scheduler, epoch, iteration, tracker, grad_scaler]

        model_only: 1st priority, force keys to [network]
        load_keys or ignore_keys: 2nd priority, exclusive. both non-None is prohibited.

        """
        if (load_keys is not None) and (ignore_keys is not None):
            raise ValueError(f"[ERROR:TRAINER] Load state dict load_keys and ignore_keys cannot be both activated.")

        default_keys = ["network", "optimizer", "scheduler", "epoch", "iteration", "tracker", "grad_scaler"]

        if model_only:
            load_keys = ["network"]
        elif load_keys is not None:
            load_keys = [k.lower() for k in load_keys]
        elif ignore_keys is not None:
            ignore_keys = [k.lower() for k in ignore_keys]
            load_keys = [k for k in default_keys if k not in ignore_keys]
        else:  # both are None
            load_keys = default_keys

        if "network" in load_keys:
            self.model.load_state_dict(state_dict["network"], strict=strict)
            print_log(f"[LOG:TRAINER] Load state dict 'network' (strict={strict}).")
        else:
            print_log(f"[WARN:TRAINER] Load state dict does not contain 'network' as key.")

        if "epoch" in load_keys:
            self.current_epoch = state_dict.get("epoch", 0)
            print_log(f"[LOG:TRAINER] Load state dict 'epoch' (current_epoch: {self.current_epoch}).")
        if "iteration" in load_keys:
            self.current_iteration = state_dict.get("iteration", 0)
            print_log(f"[LOG:TRAINER] Load state dict 'iteration' (current_iteration: {self.current_iteration}).")

        if ("optimizer" in load_keys) and (self.optimizer is not None):
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])
                print_log(f"[LOG:TRAINER] Load state dict 'optimizer'.")
            else:
                print_log(f"[LOG:TRAINER] Tried to load state dict 'optimizer', but it does not exist.")

        if ("scheduler" in load_keys) and (self.scheduler is not None):
            if "scheduler" in state_dict:
                self.scheduler.load_state_dict(state_dict["scheduler"])
                print_log(f"[LOG:TRAINER] Load state dict 'scheduler'.")
            else:
                print_log(f"[LOG:TRAINER] Tried to load state dict 'scheduler', but it does not exist.")

        if ("grad_scaler" in load_keys) and (self.grad_scaler is not None):
            if self.fp16 and ("grad_scaler" in state_dict):
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
                print_log(f"[LOG:TRAINER] Load state dict 'grad_scaler'.")

        if "tracker" in load_keys:
            if "tracker" in state_dict:
                self.common_tracker.load_state_dict(state_dict["tracker"]["common"])
                self.train_tracker.load_state_dict(state_dict["tracker"]["train"])
                self.valid_tracker.load_state_dict(state_dict["tracker"]["valid"])
                self.test_tracker.load_state_dict(state_dict["tracker"]["test"])

        self._state_loaded = True

    @staticmethod
    def distributed_setup(gpus: int, **kwargs) -> Tuple[str, int, int, bool]:
        """Setup DDP if required, returns (device, rank, world_size, is_distributed)"""
        if dist.is_available() and dist.is_initialized():  # already called before
            return f"cuda:{torch.cuda.current_device()}", get_rank(), get_world_size(), True

        if gpus <= 0:
            return "cpu", 0, 1, False
        if gpus == 1:
            torch.cuda.set_device(0)
            return "cuda:0", 0, 1, False

        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("[ERROR:TRAINER] More than 1 gpus used but LOCAL_RANK is not set. "
                               "Use torch.distributed.launch with --use_env, or use phd_torch.launch.")
        if "WORLD_SIZE" in os.environ:
            if int(os.environ["WORLD_SIZE"]) > torch.cuda.device_count():
                raise RuntimeError("[ERROR:TRAINER] GPU size is larger than available GPUs in this machine. "
                                   "Currently multi-node training is not supported.")

        ok = init_distributed(cuda=True, **kwargs)
        assert ok
        local_rank = get_rank()
        world_size = get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return device, local_rank, world_size, ok

    def resume(self, config: Dict[str, Any]) -> bool:
        # return true if resume happened.

        if "resume" in config:  # check if config is the outer-most.
            config = copy.deepcopy(config["resume"])

        if config["from_scratch"] and (config["checkpoint"] is None):
            return False

        # start from checkpoint
        if config["checkpoint"] is None:
            raise ValueError(f"[ERROR:RESUME] Resume ON, but checkpoint is None.")

        self.load_state_dict(
            state_dict=torch.load(config["checkpoint"], map_location=self.device),
            strict=config.get("strict", True),
            model_only=config.get("model_only", False),
            load_keys=config.get("load_keys", None),
            ignore_keys=config.get("ignore_keys", None),
        )

        # force set LR for optimizer
        flr = config.get("force_lr", None)
        if flr is not None:
            flr = float(flr)
            if self.optimizer is None:
                raise ValueError("[ERROR:RESUME] force_lr ON, but trainer optimizer is None.")
            self.optimizer.force_lr(flr)

        return True
