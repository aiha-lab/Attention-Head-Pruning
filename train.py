from typing import Optional, Dict, List, Tuple
import platform
import argparse
import os
import wandb
import json
import pprint
import math
import torch
import torch.cuda.amp as amp

from happy_torch.data.datasets import build_dataset
from happy_torch.data.text import LMIterator
from happy_torch.data.transforms import WordAugment

from all_transformer_xl import AllTransformerLM
from happy_torch.nn import HappyDistributedDataParallel
from happy_torch.optim import build_optimizer
from happy_torch.optim.lr_scheduler import build_scheduler
from happy_torch.trainer import BaseTrainer, BaseCompositeModel
from happy_torch.utils.dist_utils import (init_distributed, is_master, get_world_size, get_rank,
                                          split_dataloader_config, all_reduce_tensor)
from happy_torch.utils.param_utils import compute_param_norm, print_params, nan_exist
from happy_torch.utils.print_utils import time_log, print_log
from happy_torch.utils.seed_utils import torch_init
from happy_torch.utils.config_utils import override_config, override_config_by_key_value
from happy_torch.utils.tracker import MetricTrackerDict


class LMCompositeModel(BaseCompositeModel):

    def __init__(self, network, losses=None, loss_coefficients=None, metrics=None):
        super(LMCompositeModel, self).__init__(network, losses, loss_coefficients, metrics)

    def forward(self, input_: torch.Tensor, target_: Optional[torch.Tensor],
                     memory_: Optional[List[torch.Tensor]] = None
                     ) -> Tuple[Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        output = dict()
        if target_ is None:
            prob_, _, _, new_memory = self.network(input_, target_, memory_)
            output['prob'] = prob_
        else:
            ce_loss, gate_loss, gate_sparsity, new_memory = self.network(input_, target_, memory_)
            output['loss'] = ce_loss
            output['l0_loss'] = gate_loss
            output['sparsity'] = gate_sparsity[0] / gate_sparsity[1]
        return output, new_memory


class LMTrainer(BaseTrainer):

    def __init__(self, *args, length_config: dict,
                 l0_coefficient: float, l0_start_epoch: int,
                 **kwargs):
        super(LMTrainer, self).__init__(*args, **kwargs)
        self.train_metric_tracker = MetricTrackerDict('loss', 'l0_loss', 'sparsity')
        self.valid_metric_tracker = MetricTrackerDict('loss', 'l0_loss', 'sparsity')
        self.length_config = length_config
        self.l0_coefficient = l0_coefficient
        self.l0_start_epoch = l0_start_epoch

    def set_length(self, mode: str = 'train'):
        if self.is_distributed:
            self.model.module.network.set_length(*self.length_config[mode])
        else:
            self.model.network.set_length(*self.length_config[mode])

    def set_same_length(self, flag: bool = False):
        if self.is_distributed:
            self.model.module.network.set_same_length(flag)
        else:
            self.model.network.set_same_length(flag)

    def train_epoch_body(self):
        self.model.train()
        self.train_metric_tracker.reset()
        self.set_length('train')
        # self.set_same_length(False)
        self.set_same_length(True)

        self.train_dataloader: LMIterator
        self.train_dataloader.shuffle()

        with torch.enable_grad():
            memory = None
            for train_iter, (train_input, train_label, train_seq_len) in enumerate(self.train_dataloader):

                if self.current_epoch < self.l0_start_epoch:
                    l0_coeff = self.l0_coefficient * (self.current_epoch + 1) / self.l0_start_epoch
                else:
                    l0_coeff = self.l0_coefficient

                # ---------------------------------------------- #
                train_input = train_input.to(self.device, non_blocking=True)
                train_label = train_label.to(self.device, non_blocking=True)
                batch_size, seq_len = train_input.shape
                assert train_input.shape == train_label.shape
                #
                self.time_tracker.update('data')
                # ---------------------------------------------- #
                self.time_tracker.reset('forward')
                self.optimizer.zero_grad(set_to_none=True)
                grad_scale = None

                if not self.is_fp16:
                    train_output, memory = self.model(train_input, train_label, memory)
                    loss = train_output['loss']
                    l0_loss = train_output['l0_loss']
                    sparsity = train_output['sparsity']
                    #
                    self.time_tracker.update('forward')

                    # ---------------------------------------------- #
                    self.time_tracker.reset('backward')
                    #
                    loss_sum = loss + l0_loss * l0_coeff
                    # if nan_exist(loss_sum):
                    #     continue
                    loss_sum.backward()
                    #
                    self.time_tracker.update('backward')
                    # ---------------------------------------------- #
                    grad_norm = torch.as_tensor(0, )
                    if self.clip_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    param_norm = compute_param_norm(self.model.parameters())

                    # ---------------------------------------------- #
                    self.time_tracker.reset('optimizer')
                    #
                    self.optimizer.step()
                    self.scheduler.step()
                    #
                    self.time_tracker.update('optimizer')
                    # ---------------------------------------------- #
                else:  # fp16
                    with amp.autocast():
                        train_output, memory = self.model(train_input, train_label, memory)
                        loss = train_output['loss']
                        l0_loss = train_output['l0_loss']
                        sparsity = train_output['sparsity']
                        #
                        self.time_tracker.update('forward')

                        # ---------------------------------------------- #
                        self.time_tracker.reset('backward')
                        #
                        loss_sum = loss + l0_loss * l0_coeff
                        # if nan_exist(loss_sum):
                        #     continue
                        self.grad_scaler.scale(loss_sum).backward()
                        #
                        self.time_tracker.update('backward')
                        # ---------------------------------------------- #
                        self.grad_scaler.unscale_(self.optimizer)
                        grad_norm = torch.as_tensor(0, )
                        if self.clip_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        param_norm = compute_param_norm(self.model.parameters())

                        # ---------------------------------------------- #
                        self.time_tracker.reset('optimizer')
                        #
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.scheduler.step()
                        grad_scale = self.grad_scaler.get_scale()
                        #
                        self.time_tracker.update('optimizer')
                        # ---------------------------------------------- #

                if self.is_distributed:
                    loss = all_reduce_tensor(loss, 'mean')
                    # We don't have to reduce l0_loss or sparsity. Value should be same through devices.
                    batch_size *= get_world_size()

                self.train_metric_tracker.update_add({
                    'loss': (loss * seq_len * batch_size, seq_len * batch_size),
                    'l0_loss': l0_loss,
                    'sparsity': sparsity
                })

                if is_master():
                    if (train_iter % self.print_interval == 0) and (train_iter > 0):
                        s = time_log() + '\n'
                        s += f'...... Iter {train_iter} / {len(self.train_dataloader)} ' \
                             f'(epoch: {self.current_epoch} / {self.max_epochs})\n'
                        s += f'......... loss(batch/avg): ' \
                             f'{loss.item():.6f} / ' \
                             f'{self.train_metric_tracker.average("loss"):.6f}\n' \
                             f'......... ppl(batch/avg): ' \
                             f'{math.exp(loss.item()):.6f} / ' \
                             f'{math.exp(self.train_metric_tracker.average("loss")):.6f}\n' \
                             f'......... l0 loss(batch/avg), (coefficient: {l0_coeff:.3f}/{self.l0_coefficient:.3f}): ' \
                             f'{l0_loss.item():.6f} / ' \
                             f'{self.train_metric_tracker.average("l0_loss"):.6f}\n' \
                             f'......... sparsity(batch/avg): ' \
                             f'{sparsity.item():.6f} / ' \
                             f'{self.train_metric_tracker.average("sparsity"):.6f}\n' \
                             f'......... grad/param norm: ' \
                             f'{grad_norm.item():.6f} / ' \
                             f'{param_norm.item():.6f}\n' \
                             f'......... data/forward/backward/optimizer time: ' \
                             f'{self.time_tracker.get("data"):.4f} / ' \
                             f'{self.time_tracker.get("forward"):.4f} / ' \
                             f'{self.time_tracker.get("backward"):.4f} / ' \
                             f'{self.time_tracker.get("optimizer"):.4f}\n'
                        s += f'...... LR: {self.optimizer.current_lrs()[0]:.6f}\n' \
                             f'...... Sequence length: {seq_len}, Batch size: {batch_size} ' \
                             f'(per-gpu: {batch_size // get_world_size()})'
                        if grad_scale is not None:
                            s += f'\n...... Gradient scaler: {grad_scale:.4f}'
                        self.print(s)

                    if self.current_iteration % self.log_interval == 0:
                        wandb.log({
                            'train_loss': loss.item(),
                            'train_ppl': math.exp(loss.item()),
                            'train_l0_loss': l0_loss.item(),
                            'train_sparsity': sparsity.item(),
                            'grad_norm': grad_norm.item(),
                            'param_norm': param_norm.item(),
                            'lr': self.optimizer.current_lrs()[0],
                            'iterations': self.current_iteration,
                            'tokens': (self.current_iteration + 1) * seq_len * batch_size,
                        })

                # ---------------------------------------------- #
                if (0 < self.valid_interval < 1) and (
                        train_iter % int(self.valid_interval * len(self.train_dataloader)) == 0) and (
                        train_iter > 0):
                    self.valid()

                    # restart!
                    self.model.train()
                    self.train_metric_tracker.reset()

                # ---------------------------------------------- #
                self.time_tracker.reset('data')
                self.current_iteration += 1

        if is_master():
            save_path = os.path.join(wandb.run.dir, 'latest.ckpt')
            self.print(f'Save LATEST state_dict to {save_path}')
            torch.save(self.state_dict(), save_path)

    def valid_body(self, *, track: bool = True):
        self.model.eval()
        self.valid_metric_tracker.reset()
        self.set_length('valid')
        # self.set_same_length(False)
        self.set_same_length(True)

        with torch.no_grad():
            memory = None
            for valid_iter, (valid_input, valid_label, valid_seq_len) in enumerate(self.valid_dataloader):
                # ---------------------------------------------- #
                valid_input = valid_input.to(self.device, non_blocking=True)
                valid_label = valid_label.to(self.device, non_blocking=True)
                batch_size, seq_len = valid_input.shape
                assert valid_input.shape == valid_label.shape

                # ---------------------------------------------- #
                valid_output, memory = self.model(valid_input, valid_label, memory)
                loss = valid_output['loss']
                l0_loss = valid_output['l0_loss']
                sparsity = valid_output['sparsity']

                # ---------------------------------------------- #
                if self.is_distributed:
                    loss = all_reduce_tensor(loss, 'mean')
                    batch_size *= get_world_size()

                self.valid_metric_tracker.update_add({
                    'loss': (loss * seq_len * batch_size, seq_len * batch_size),
                    'l0_loss': l0_loss,
                    'sparsity': sparsity,
                })

        valid_loss = self.valid_metric_tracker.average("loss")
        valid_l0_loss = self.valid_metric_tracker.average("l0_loss")
        valid_sparsity = self.valid_metric_tracker.average("sparsity")

        if is_master():
            s = time_log() + '\n'
            s += f'...... valid epoch loss(avg): ' \
                 f'{valid_loss:.6f}\n' \
                 f'...... valid epoch ppl(avg): ' \
                 f'{math.exp(valid_loss):.6f}\n' \
                 f'...... valid epoch l0 loss(avg): ' \
                 f'{valid_l0_loss:.6f} \n' \
                 f'...... valid epoch sparsity(avg): ' \
                 f'{valid_sparsity:.6f}'
            self.print(s)

            wandb.log({
                'valid_loss': valid_loss,
                'valid_ppl': math.exp(valid_loss),
                'valid_l0_loss': valid_l0_loss,
                'valid_sparsity': valid_sparsity,
                'epoch': self.current_epoch,
                'iterations': self.current_iteration,
            })

        is_updated = self.scheduler.update_best(valid_loss)
        if is_master() and is_updated:
            wandb.run.summary["best_loss"] = valid_loss
            wandb.run.summary["best_epoch"] = self.current_epoch

            save_path = os.path.join(wandb.run.dir, 'best.ckpt')
            self.print(f'Save BEST state_dict to {save_path}')
            torch.save(self.state_dict(), save_path)
        elif is_master() and (self.current_epoch % 20 == 19):
            save_path = os.path.join(wandb.run.dir, f'epoch_{self.current_epoch}.pth')
            self.print(f'Save Epoch {self.current_epoch} state_dict to {save_path}')
            torch.save(self.state_dict(), save_path)

    def test_body(self):
        self.model.eval()
        self.valid_metric_tracker.reset()  # reuse
        self.set_length('test')
        self.set_same_length(True)

        with torch.no_grad():
            memory = None
            for test_iter, (test_input, test_label, test_seq_len) in enumerate(self.test_dataloader):
                # ---------------------------------------------- #
                test_input = test_input.to(self.device, non_blocking=True)
                test_label = test_label.to(self.device, non_blocking=True)
                batch_size, seq_len = test_input.shape
                assert test_input.shape == test_label.shape

                # ---------------------------------------------- #
                test_output, memory = self.model(test_input, test_label, memory)
                loss = test_output['loss']
                l0_loss = test_output['l0_loss']
                sparsity = test_output['sparsity']

                # ---------------------------------------------- #
                if self.is_distributed:
                    loss = all_reduce_tensor(loss, 'mean')
                    batch_size *= get_world_size()

                self.valid_metric_tracker.update_add({
                    'loss': (loss * seq_len * batch_size, seq_len * batch_size),
                    'l0_loss': l0_loss,
                    'sparsity': sparsity,
                })

        test_loss = self.valid_metric_tracker.average("loss")
        test_l0_loss = self.valid_metric_tracker.average("l0_loss")
        test_sparsity = self.valid_metric_tracker.average("sparsity")

        if is_master():
            s = time_log() + '\n'
            s += f'...... Test epoch loss(avg): ' \
                 f'{test_loss:.6f}\n' \
                 f'...... Test epoch ppl(avg): ' \
                 f'{math.exp(test_loss):.6f}\n' \
                 f'...... Test epoch l0 loss(avg): ' \
                 f'{test_l0_loss:.6f} \n' \
                 f'...... Test epoch sparsity(avg): ' \
                 f'{test_sparsity:.6f}'
            self.print(s)

            wandb.log({
                'test_loss': test_loss,
                'test_ppl': math.exp(test_loss),
                'test_l0_loss': test_l0_loss,
                'test_sparsity': test_sparsity,
                'epoch': self.current_epoch,
                'iterations': self.current_iteration,
            })


def run(cfg: Dict):
    # ======================================================================================== #
    # Distributed
    # ======================================================================================== #
    num_gpus = cfg['gpus']
    if num_gpus > 1:
        if 'LOCAL_RANK' not in os.environ:
            raise RuntimeError('More than 1 gpus used but LOCAL_RANK is not set. '
                               'Use torch.distributed.launch with --use_env.')
        if 'WORLD_SIZE' in os.environ:
            if int(os.environ['WORLD_SIZE']) > torch.cuda.device_count():
                raise RuntimeError('GPU size is larger than available GPUs in this machine. '
                                   'Currently multi-node training is not yet tested.')
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()

    if is_distributed and num_gpus <= 1:
        raise RuntimeError('Distributed ON but only 1 gpu is set. '
                           'Consider NOT using torch.distributed.launch for single gpu or cpu.')

    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}' if (num_gpus > 0) else 'cpu'

    print(f'START: {local_rank} / {world_size}')

    # ======================================================================================== #
    # Init
    # ======================================================================================== #
    torch_init(cfg['seed'] + local_rank)  # different seed for different GPUs
    run_type = cfg['run_type']
    save_dir = cfg['save_dir']
    if is_master():
        os.makedirs(save_dir, exist_ok=True)

    # ======================================================================================== #
    # Log
    # ======================================================================================== #
    if is_master():
        assert is_master()
        wandb_mode = cfg['wandb']['mode']
        wandb_project = cfg['project']
        wandb_name = cfg['name']
        server_name = platform.node()
        wandb_note = cfg['wandb']['notes']
        wandb_note = server_name + (f'-{wandb_note}' if (wandb_note is not None) else '')

        wandb.init(project=wandb_project, job_type=run_type, name=wandb_name, dir=save_dir,
                   mode=wandb_mode, notes=wandb_note, config=cfg)
        pprint.pprint(cfg)

    # ======================================================================================== #
    # Distributed
    # ======================================================================================== #
    if is_distributed:
        cfg = split_dataloader_config(cfg, world_size, local_rank)

    # ======================================================================================== #
    # Create Dataset
    # ======================================================================================== #
    train_dataset = build_dataset(cfg['dataset']['train_dataset'])
    valid_dataset = build_dataset(cfg['dataset']['valid_dataset'])
    test_dataset = build_dataset(cfg['dataset']['test_dataset'])

    vocab_size = train_dataset.vocab_size

    # ======================================================================================== #
    # Create DataLoader
    # ======================================================================================== #
    train_dataloader = LMIterator.from_config(cfg['dataloader']['train_dataloader'], train_dataset.data,
                                              local_rank, world_size)
    valid_dataloader = LMIterator.from_config(cfg['dataloader']['valid_dataloader'], valid_dataset.data,
                                              local_rank, world_size)
    test_dataloader = LMIterator.from_config(cfg['dataloader']['test_dataloader'], test_dataset.data,
                                             local_rank, world_size)

    # currently don't recommend to use
    train_transform = WordAugment(vocab_size, cfg['augment']['num_swap'], cfg['augment']['num_replace'])
    train_dataloader.transform = train_transform

    length_config = dict()
    length_config['train'] = (cfg['dataloader']['train_dataloader']['seq_length'],
                              cfg['dataloader']['train_dataloader']['mem_length'],
                              cfg['dataloader']['train_dataloader']['overlap_length'])
    length_config['valid'] = (cfg['dataloader']['valid_dataloader']['seq_length'],
                              cfg['dataloader']['valid_dataloader']['mem_length'],
                              cfg['dataloader']['valid_dataloader']['overlap_length'])
    length_config['test'] = (cfg['dataloader']['test_dataloader']['seq_length'],
                             cfg['dataloader']['test_dataloader']['mem_length'],
                             cfg['dataloader']['test_dataloader']['overlap_length'])

    # ======================================================================================== #
    # Create Network
    # ======================================================================================== #
    # intial setting with train length
    network = AllTransformerLM.from_config(cfg['model'], vocab_size, *length_config['test'])
    print_params(network)
    model = LMCompositeModel(network).to(device)

    if is_distributed:
        print_log('USING Native Pytorch DDP')
        model = HappyDistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                             find_unused_parameters=True)

    # ======================================================================================== #
    # Create Optimizer and Scheduler
    # ======================================================================================== #
    non_decay_params = []
    decay_params = []
    for param_name, param_val in network.named_parameters():
        if ('norm' in param_name) or ('gate' in param_name):
            non_decay_params.append(param_val)
        else:
            decay_params.append(param_val)

    params_for_opt = [
        {'params': decay_params},
        {'params': non_decay_params, 'weight_decay': 0.0}
    ]

    # optimizer = build_optimizer(cfg['optimizer'], network.parameters())
    optimizer = build_optimizer(cfg['optimizer'], params_for_opt)
    scheduler = build_scheduler(cfg['scheduler'], optimizer)

    # ======================================================================================== #
    # Set Trainer
    # ======================================================================================== #
    trainer = LMTrainer(
        model=model,
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        length_config=length_config,
        l0_coefficient=cfg['custom']['l0_coefficient'],
        l0_start_epoch=cfg['custom']['l0_start_epoch'],
        **cfg['trainer']  # max_epochs, print_interval, log_interval, valid_interval, clip_grad_norm, fp16
    )
    if cfg['resume']:
        if cfg['checkpoint'] is None:
            cfg['checkpoint'] = os.path.join(wandb.run.dir, 'latest.ckpt')
        # trainer.load_state_dict(torch.load(cfg['checkpoint'], map_location=device), model_only=True)

        # ad-hoc fix
        old_ckpt = torch.load(cfg['checkpoint'], map_location=device)
        new_ckpt = {}
        new_ckpt['network'] = {}
        for k, v in old_ckpt['network'].items():
            new_k = k.replace('module.', 'module.network.')
            new_ckpt['network'][new_k] = v
        trainer.load_state_dict(new_ckpt, model_only=True)

    # ======================================================================================== #
    # Run
    # ======================================================================================== #
    if run_type == 'train':
        trainer.train()
    elif run_type == 'test':
        trainer.test()
    else:
        raise ValueError(f'Config run_type should be either train or test, got {run_type}')

    if is_master():
        wandb.run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default config json
    parser.add_argument('--config', type=str, help='Run configuration', required=True)
    # override config by json
    parser.add_argument('--override', default=None, type=str, help='Override configuration')
    # override by command line (highest priority)
    parser.add_argument('--data_dir', default=None, type=str, help='Data directory')
    parser.add_argument('--run_type', default=None, type=str, help='Run type')
    parser.add_argument('--gpus', default=None, type=int, help='Number of GPUs to use')

    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume flag')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint load path')
    parser.add_argument('--fp16', action='store_true', help='Using FP16')

    parser.add_argument('--notes', default=None, type=str, help='Wandb note')
    parser.add_argument('--wandb_offline', action='store_true', help='Offline wandb')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')
    args = parser.parse_args()

    # ------------------------------------------------ #
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.override is not None:
        with open(args.override, 'r') as f:
            override = json.load(f)
        config = override_config(config, override)

    if args.data_dir is not None:
        override_config_by_key_value(config, 'data_dir', args.data_dir)
    if args.run_type is not None:
        config['run_type'] = args.run_type
    if args.gpus is not None:
        config['gpus'] = args.gpus
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
    if args.batch_size is not None:
        config['dataloader']['train_dataloader']['batch_size'] = args.batch_size
        config['dataloader']['valid_dataloader']['batch_size'] = args.batch_size
        config['dataloader']['test_dataloader']['batch_size'] = args.batch_size
    if args.fp16:
        config['trainer']['fp16'] = args.fp16
    if args.resume:
        config['resume'] = args.resume
    if args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint

    if args.notes is not None:
        config['wandb']['notes'] = args.notes
    if args.wandb_offline:
        config['wandb']['mode'] = 'offline'
    if args.wandb_disabled:
        config['wandb']['mode'] = 'disabled'

    # ------------------------------------------------ #
    run(config)
