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

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# decided to do distributed training from here instead of relying on misc
import torch.distributed as dist 
import torch.multiprocessing as mp

import models_mae
from multi_head_mae import MultiHeadMAE

from engine_pretrain import train_one_epoch
from dataset_generation.emoji_dataset import EmojiDataset

import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
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

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_name', default='cifar', type=str,
                        help='name of dataset, either cifar, emoji, or imagenet')
    parser.add_argument('--image_keywords', default=None, nargs='+',
                        help='Categories of emojis you want')
    
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

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
                    

    return parser

def main_distributed(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', init_method=args.dist_url, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank + 4}")  # Use the last 4 GPUs (4, 5, 6, 7)
    torch.cuda.set_device(device)  # Set the current device for this process
    seed = args.seed + rank

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.amp.autocast(enabled=False)
    cudnn.benchmark = True

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train) if args.dataset_name == 'cifar' else datasets.ImageNet(args.data_path, split="train", transform=transform_train, is_valid_file=lambda x: not x.split('/')[-1].startswith('.'))
    # dataset_val = datasets.CIFAR100('../data', train=False, download=True, transform=transform_val)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    
    if rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    if (args.lower is not None) and (args.median is not None) and (args.upper is not None):
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
                                                quantile=args.quantile)
        print('create point model')

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    # print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * world_size
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank + 4])
    model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if (args.lower and args.median and args.upper):
        wandb_name = f'multiDecoder_{args.lower}_{args.median}_{args.upper}'
    elif args.quantile:
        wandb_name = f'quantile_{args.quantile}'
    else:
        wandb_name = f'mse'

    model_name = wandb_name + datetime.datetime.now().strftime("%H:%M:%S")
    wandb.init(config=args, project='pretrain_mae', name=f"model_{model_name}")
    wandb.watch(model)
    if rank == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            1.0, # Added this part
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.log_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        wandb.log(log_stats, step=epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    from thop import profile, clever_format  # run pip3 install thop for this, i don't think it was in the original requirements.txt

    if rank == 0:
        dummy_input = torch.rand(1, 3, 224, 224).to(device)  # should give a good estimate of the model's complexity
        flops, params = profile(model, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {flops}, Params: {params}")

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                       **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters, 
                        'flops': flops,
                        'runtime': total_time_str}
    wandb.log(log_stats)


def main(args):
    
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
    # transform_val = transforms.Compose([
    #         transforms.Resize(256, interpolation=3),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    if args.dataset_name == 'cifar':
        dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
    elif args.dataset_name == 'emoji':
        dataset_train = EmojiDataset(os.path.join(args.data_path, 'train'), args.image_keywords)
    else:
        dataset_train = datasets.ImageNet(args.data_path, split="train", 
            transform=transform_train, is_valid_file=lambda x: not x.split('/')[-1].startswith('.'))
    # dataset_val = datasets.CIFAR100('../data', train=False, download=True, transform=transform_val)

    
    # print(dataset_train[0][0].shape)

    # train_indices = [i for i in range(45000)]
    # print(f'using only {len(train_indices)} indices')
    # dataset_train = torch.utils.data.Subset(dataset_train, train_indices)

    
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(train_sampler))
    # val_sampler = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    # why no validation?
    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False
    # )
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    if (args.lower is not None) and (args.median is not None) and (args.upper is not None):
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
                                                quantile=args.quantile)
        print('create point model')

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if (args.lower and args.median and args.upper):
        wandb_name = f'multiDecoder_{args.lower}_{args.median}_{args.upper}'
    elif args.quantile:
        wandb_name = f'quantile_{args.quantile}'
    else:
        wandb_name = f'mse'

    wandb.init(config=args, project='pretrain_mae', name=f"model_{wandb_name}")
    wandb.watch(model)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            5, # Added this part
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 40 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        wandb.log(log_stats, step=epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Distributed?", args.distributed)
    if args.distributed:
        world_size = 4  # Use only the last 4 GPUs
        mp.spawn(main_distributed, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(args)
