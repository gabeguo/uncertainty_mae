import argparse
import datetime
import json
import numpy as np
import random
import os
import time
from pathlib import Path
from functools import partial

from datasets import load_dataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# decided to do distributed training from here instead of relying on misc
import torch.distributed as dist 
import torch.multiprocessing as mp

import models_mae
from multi_head_mae import MultiHeadMAE
from uncertainty_mae import UncertaintyMAE

from engine_pretrain import train_one_epoch, create_discriminator
from dataset_generation.emoji_dataset import EmojiDataset

import coco_transforms

import wandb

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

# Thanks https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html

def ddp_setup(rank, world_size, master_port="12355"):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader_train: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        loss_scaler,
        log_writer,
        model_without_ddp,
        args
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.data_loader_train = data_loader_train
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.log_writer = log_writer
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=isinstance(model, UncertaintyMAE))
        self.model_without_ddp = model_without_ddp
        self.args = args

        if args.gan:
            # TODO: add weight decay?
            # optim_factory.add_weight_decay(model.visible_mae, args.weight_decay)
            self.netD = create_discriminator(args, self.model)
            self.netD = DDP(self.netD, device_ids=[gpu_id])
            self.optimizerD = torch.optim.AdamW(self.netD.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
            self.optimizerG = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.95), eps=args.eps)
        else:
            self.netD = None
            self.optimizerD = None
            self.optimizerG = None

        return

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.data_loader_train)}")
        self.data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=self.model, data_loader=self.data_loader_train,
            optimizer=self.optimizer, device=self.gpu_id, epoch=epoch, loss_scaler=self.loss_scaler,
            max_norm=5, # Added this part
            log_writer=self.log_writer,
            args=self.args,
            netD=self.netD, optimizerD=self.optimizerD, optimizerG=self.optimizerG
        )

        if self.gpu_id == 0:
            if self.args.output_dir and (epoch % self.args.log_freq == 0 or epoch + 1 == self.args.epochs):
                misc.save_model(
                    args=self.args, model=self.model.module, model_without_ddp=self.model_without_ddp, 
                    optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch=epoch)

            # TODO: may be collective call??
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}
            wandb.log(log_stats, step=epoch)

            if self.args.output_dir and misc.is_main_process():
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self._run_epoch(epoch)
            if epoch == self.args.kill_epoch:
                return
        return


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--partial_vae', action='store_true',
                        help='Whether to use a regular MAE on the visible patches, and VAE on invisible patches')
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--quantile', default=None, type=float,
                        help='None if we train with MSE. Otherwise, set to # between 0 and 1 for pinball loss.')
    parser.add_argument('--lower', default=None, type=float,
                        help='Lower quantile for multi-head decoder')
    parser.add_argument('--median', default=None, type=float,
                        help='Median for multi-head decoder')
    parser.add_argument('--upper', default=None, type=float,
                        help='Upper quantile for multi-head decoder')
    parser.add_argument('--vae', action='store_true', 
                        help='is this a vae?')
    parser.add_argument('--num_vae_blocks', default=1, type=int,
                        help='number of VAE blocks (for mean and var) to add on top of backbone')
    parser.add_argument('--kld_beta', default=1, type=float,
                        help='Beta term if using VAE')
    parser.add_argument('--dropout_ratio', default=0, type=float,
                        help='How often to ignore the invisible encoder')
    parser.add_argument('--same_encoder', action='store_true',
                        help='do we use same encoder for visible and invisible?')
    parser.add_argument('--end_to_end_finetune', action='store_true',
                        help='are we end-to-end finetuning the loaded pretrained_weights?')
    parser.add_argument('--block_mask_prob', default=0, type=float,
                        help='What probability to use a contiguous mask instead of random mask?')
    parser.add_argument('--object_mask', action='store_true',
                        help='On Coco, do we use semantic masks?')
    parser.add_argument('--add_default_mask', action='store_true',
                        help='Do we add default mask?')
    parser.add_argument('--disable_zero_conv', action='store_true',
                        help='Disable zero conv to initialize VAE?')
    parser.add_argument('--var', default=1, type=float,
                        help='The var to use for KLD loss (assume mean 0)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='AdamW epsilon (default: 1e-8)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Whether to mix between fp16 and fp32')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--invisible_lr_scale', type=float, default=None, metavar='LR',
                        help='multiplicative factor to scale invisible LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_name', default='cifar', type=str,
                        help='name of dataset, either cifar, emoji, or imagenet')
    parser.add_argument('--include_keywords', default=None, nargs='+',
                        help='Categories of emojis you do want')
    parser.add_argument('--exclude_keywords', default=None, nargs='+',
                        help='Categories of emojis you dont want')
    parser.add_argument('--include_any', action='store_true',
                        help='do we include images that have ANY of the keywords, or ALL?')
    parser.add_argument('--exclude_any', action='store_true',
                        help='do we exclude images that have ANY of the keywords, or ALL?')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--log_freq', default=40, type=int,
                        help='how many epochs in between logs')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='weights from previous MAE')
    parser.add_argument('--frozen_backbone_epochs', type=int, default=None,
                        help='How many epochs to keep pre-trained layers frozen for')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--kill_epoch', default=801, type=int,
                        help='epoch to kill it on')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='do distributed training or no distributed training')
    
    # my distributed training parameters
    parser.add_argument('--master_port', default="12355", type=str)
    
    # logging parameters
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='pretrain_mae_new')
                    
    # GAN params
    parser.add_argument('--gan', action='store_true',
                        help='Whether to do GAN training')
    parser.add_argument('--discriminator_drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--discriminator_global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    return parser

def main(rank, args, world_size):
    
    ddp_setup(rank, world_size, args.master_port)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.amp.autocast(enabled=False)
    cudnn.benchmark = True

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    # TODO: better transform
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if args.dataset_name == 'cifar':
        dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
    elif args.dataset_name == 'celeba':
        transform_celeba = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_train = datasets.CelebA('/local/zemel/gzg2104/datasets', split='train', target_type='attr', transform=transform_celeba, download=True)
    elif args.dataset_name == 'flowers':
        dataset_train = datasets.Flowers102('../data', split='train', transform=transform_train, download=True)
    elif args.dataset_name == 'food':
        dataset_train = datasets.Food101('/local/zemel/gzg2104/datasets', split='train', transform=transform_train, download=True)
    elif args.dataset_name == 'emoji':
        dataset_train = EmojiDataset(args.data_path, include_keywords=args.include_keywords, exclude_keywords=args.exclude_keywords,
                                     include_any=args.include_any, exclude_any=args.exclude_any)
    elif args.dataset_name == 'emnist':
        emnist_mean = np.array([0.176, 0.176, 0.176])
        emnist_std = np.array([0.328, 0.328, 0.328])
        emnist_transform = transforms.Compose([
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(emnist_mean, emnist_std)
            ])
        dataset_train = datasets.EMNIST('../data', split='balanced', 
                                        train=True, download=True,
                                        transform=emnist_transform)
    elif args.dataset_name == 'imagenet_sketch':
        sketch_mean = [0.857, 0.857, 0.857]
        sketch_std = [0.254, 0.254, 0.254]
        transform_sketch = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                # transforms.Resize((224, 224), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(sketch_mean, sketch_std)
            ])
        def transform_wrapper(examples):
            examples["image"] = [transform_sketch(image) for image in examples["image"]]
            return examples

        dataset_train = load_dataset("imagenet_sketch", split='train', 
                            cache_dir='/local/zemel/gzg2104/datasets')

        dataset_train.set_transform(transform_wrapper)
    elif args.dataset_name == 'coco':
        ds = load_dataset("detection-datasets/coco")
        dataset_train = ds['train']
        
        dataset_train.set_transform(partial(coco_transforms.transform_function, mask_ratio=args.mask_ratio))
    else:
        print('imagenet!')
        dataset_train = datasets.ImageNet(args.data_path, split="train", 
            transform=transform_train, is_valid_file=lambda x: not x.split('/')[-1].startswith('.'))
    
        print(len(dataset_train))
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = DistributedSampler(dataset_train)
    print("Sampler_train = %s" % str(train_sampler))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=train_sampler, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        shuffle=False
    )

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    if args.partial_vae:
        visible_model = None if args.same_encoder else \
            models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                            quantile=args.quantile, vae=False, kld_beta=0)
        invisible_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                quantile=args.quantile, vae=args.vae, kld_beta=args.kld_beta,
                                                num_vae_blocks=args.num_vae_blocks, disable_zero_conv=args.disable_zero_conv)
        model = UncertaintyMAE(visible_mae=visible_model, invisible_mae=invisible_model, 
                               dropout_ratio=args.dropout_ratio,
                               load_weights=args.pretrained_weights, same_encoder=args.same_encoder,
                               end_to_end_finetune=args.end_to_end_finetune,
                               block_mask_prob=args.block_mask_prob,
                               var=args.var)
        print('partial VAE')
    elif (args.lower is not None) and (args.median is not None) and (args.upper is not None):
        assert 0 < args.lower < args.median < args.upper < 1
        lower_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                quantile=args.lower)
        median_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                quantile=args.median)
        upper_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                quantile=args.upper)
        model = MultiHeadMAE(lower_mae=lower_model, median_mae=median_model, upper_mae=upper_model)
        print('create multi-head decoder')
    else:
        assert (args.quantile is None) or (args.quantile > 0 and args.quantile < 1)
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                quantile=args.quantile, vae=args.vae, kld_beta=args.kld_beta)
        print('create point model')

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(type(model))
    if rank == 0:
        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)

    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)
    
    # following timm: set wd as 0 for bias and norm layers
    if (args.invisible_lr_scale is not None) and (not args.same_encoder):
        assert args.partial_vae
        assert args.vae
        visible_params = optim_factory.add_weight_decay(model.visible_mae, args.weight_decay)
        assert len(visible_params) == 2
        if abs(args.invisible_lr_scale) <= 1e-8:
            optimizer = torch.optim.AdamW(visible_params, 
                                        lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
            # freeze the invisible MAE!
            print('do the freezing')
            for param in model.invisible_mae.parameters():
                param.requires_grad = False
        else:
            invisible_params = optim_factory.add_weight_decay(model.invisible_mae, args.weight_decay)
            for curr_param_group in invisible_params:
                assert 'params' in curr_param_group
                assert 'weight_decay' in curr_param_group
                curr_param_group['lr_scale'] = args.invisible_lr_scale
            assert len(invisible_params) == 2
            optimizer = torch.optim.AdamW(visible_params + invisible_params, 
                                        lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if rank == 0:
        wandb_name = args.output_dir
        if args.disable_wandb:
            wandb.init(mode='disabled')
        else:
            wandb.init(config=args, 
                        project=args.wandb_project, 
                        name=wandb_name)
        wandb.watch(model)

    print(f"Start training for {args.epochs} epochs")

    trainer = Trainer(model=model, data_loader_train=data_loader_train, optimizer=optimizer,
                      gpu_id=rank, loss_scaler=loss_scaler, log_writer=log_writer,
                      model_without_ddp=model_without_ddp, args=args)
    trainer.train()

    destroy_process_group()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = get_args_parser()
    args = args.parse_args()

    world_size = torch.cuda.device_count()
    print(f"{world_size} gpus")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    mp.spawn(main, args=(args, world_size), nprocs=world_size)
