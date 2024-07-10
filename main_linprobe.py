# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate
import wandb
from multi_head_mae import MultiHeadMAE
from uncertainty_mae import UncertaintyMAE
import models_mae

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import coco_transforms
from datasets import load_dataset

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
        data_loader_test: torch.utils.data.DataLoader,
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
        self.data_loader_test = data_loader_test
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.log_writer = log_writer
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.model_without_ddp = model_without_ddp
        self.args = args

        self.criterion = torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(self.criterion))

        self.max_accuracy = 0

        return

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.data_loader_train)}")
        self.data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=self.model, criterion=self.criterion, data_loader=self.data_loader_train,
            optimizer=self.optimizer, device=self.gpu_id, epoch=epoch, loss_scaler=self.loss_scaler,
            max_norm=5, # Added this part
            log_writer=self.log_writer,
            args=self.args
        )

        test_stats = evaluate(self.data_loader_test, self.model, self.gpu_id)
        print(f"Accuracy of the network on the {len(self.data_loader_test.dataset)} test images: {test_stats['acc1']:.1f}%")
        self.max_accuracy = max(self.max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {self.max_accuracy:.2f}%')

        if self.gpu_id == 0:
            if self.args.output_dir and (epoch % self.args.log_freq == 0 or epoch + 1 == self.args.epochs):
                misc.save_model(
                    args=self.args, model=self.model.module, model_without_ddp=self.model_without_ddp, 
                    optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch=epoch)

            # TODO: may be collective call??
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        #'n_parameters': n_parameters
                        }
            wandb.log(log_stats, step=epoch)

            if self.args.output_dir and misc.is_main_process():
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        return
    
    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self._run_epoch(epoch)
        return

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_vae_blocks', default=1, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--lower_bound_model', default=None, type=str,
                        help='path to lower bound model')
    parser.add_argument('--point_bound_model', default=None, type=str,
                        help='path to point estimate model')
    parser.add_argument('--upper_bound_model', default=None, type=str,
                        help='path to upper bound model')
    parser.add_argument('--scale_factor_path', default=None, type=str,
                        help='path to scale factor')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--dataset_name', default='cifar', type=str,
                        help='name of dataset, either cifar or imagenet')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    
    parser.add_argument('--eval_weights', default='/home/gabeguo/uncertainty_mae/cifar100_linprobe_uncertainty/checkpoint-89.pth',
                        type=str, help='weights for evaluation')
    
    # my distributed training parameters
    parser.add_argument('--master_port', default="12355", type=str)

    # logging parameters
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='linprobe')

    return parser

def set_model(args, model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % weight_path)
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
        if any(['median_mae' in the_key for the_key in checkpoint_model]):
            # we have multi-head decoder
            lower_model = models_mae.__dict__['mae_' + args.model](norm_pix_loss=False, 
                                                    quantile=0.05)
            median_model = models_mae.__dict__['mae_' + args.model](norm_pix_loss=False, 
                                                    quantile=0.5)
            upper_model = models_mae.__dict__['mae_' + args.model](norm_pix_loss=False, 
                                                    quantile=0.95)
            multi_head_mae = MultiHeadMAE(lower_mae=lower_model, median_mae=median_model, upper_mae=upper_model)
            multi_head_mae.load_state_dict(checkpoint_model)
            checkpoint_model = multi_head_mae.median_mae.state_dict()
            print('load multi head MAE')
        if any(['visible_mae' in the_key for the_key in checkpoint_model]):
            visible_model = models_mae.__dict__['mae_' + args.model](vae=False)
            invisible_model = models_mae.__dict__['mae_' + args.model](vae=True,
                                    num_vae_blocks=args.num_vae_blocks, disable_zero_conv=True)
            uncertainty_mae = UncertaintyMAE(visible_mae=visible_model, invisible_mae=invisible_model)
            uncertainty_mae.load_state_dict(checkpoint_model)
            checkpoint_model = uncertainty_mae.visible_mae.state_dict()
            print('Uncertainty MAE')

    else:
        raise ValueError('model should be in checkpoint')
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    return model

def set_head(model, device):
    print(model.head.weight.size())
    print(model.head.bias.size())
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    return

def create_model(args):
    return models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

def main(rank, args, world_size):

    ddp_setup(rank, world_size, args.master_port)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

    if args.dataset_name == 'cifar':
        dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100('../data', train=False, download=True, transform=transform_val)
    # elif args.dataset_name == 'coco':
    #     ds = load_dataset("detection-datasets/coco")
    #     dataset_train = ds['train']
    #     dataset_train.set_transform(coco_transforms.transform_function)
    #     dataset_val = ds['val']
    #     dataset_val.set_transform(coco_transforms.transform_function)
    else:
        dataset_train = datasets.ImageNet(args.data_path, split="train", transform=transform_train, is_valid_file=lambda x: not x.split('/')[-1].startswith('.'))
        dataset_val = datasets.ImageNet(args.data_path, split="val", transform=transform_val, is_valid_file=lambda x: not x.split('/')[-1].startswith('.'))

    print(dataset_train)
    print(dataset_val)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train
    )
    print("Sampler_train = %s" % str(sampler_train))
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val
    )

    if rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    use_uncertainty_encoder = (args.lower_bound_model is not None) \
        and (args.upper_bound_model is not None) \
        and (args.point_bound_model is not None)

    if use_uncertainty_encoder:
        lower_model = create_model(args)
        middle_model = create_model(args)
        upper_model = create_model(args)
    else:
        model = create_model(args)

    if args.finetune and not args.eval:
        if use_uncertainty_encoder:
            print('Confidence intervals')
            lower_model = set_model(args, lower_model, args.lower_bound_model)
            middle_model = set_model(args, middle_model, args.point_bound_model)
            upper_model = set_model(args, upper_model, args.upper_bound_model)
        else:
            print('Single model')
            model = set_model(args, model, args.finetune)

    # for linear prob only
    # hack: revise model's head with BN
    set_head(model, device)
    model.cuda()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        model.load_state_dict(torch.load(args.eval_weights)['model'])
        print(f'loaded weights: {args.eval_weights}')
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

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

    trainer = Trainer(model=model, 
                      data_loader_train=data_loader_train, data_loader_test=data_loader_val,
                      optimizer=optimizer,
                      gpu_id=rank, loss_scaler=loss_scaler, log_writer=log_writer,
                      model_without_ddp=model_without_ddp, args=args)
    trainer.train()

    destroy_process_group()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"{world_size} gpus")
    mp.spawn(main, args=(args, world_size), nprocs=world_size)
