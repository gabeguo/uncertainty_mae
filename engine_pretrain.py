# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from uncertainty_mae import UncertaintyMAE


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, the_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.dataset_name == 'imagenet_sketch':
            samples = the_data['image']
        else:
            (samples, _) = the_data
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                if isinstance(model, UncertaintyMAE):
                    loss, pred, mask, reconstruction_loss, kld_loss = \
                        model(samples, mask_ratio=args.mask_ratio, return_component_losses=True)
                else:
                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        else:
            if isinstance(model, UncertaintyMAE):
                loss, pred, mask, reconstruction_loss, kld_loss = \
                    model(samples, mask_ratio=args.mask_ratio, return_component_losses=True)
            else:
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            #         print('has nan grad', any(torch.isnan(param.grad).flatten().tolist()))
            raise ValueError("Loss is {}, stopping training".format(loss_value))
        # else:
        #     print("Loss is {}, continue training".format(loss_value))

        loss /= accum_iter
        if args.mixed_precision:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss.backward()
        if (data_iter_step + 1) % accum_iter == 0:
            if (not args.mixed_precision): # loss scaler didn't weight update in full precision
                optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if isinstance(model, UncertaintyMAE):
            metric_logger.update(reconstruction_loss=reconstruction_loss)
            metric_logger.update(kld_loss=kld_loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if args.invisible_lr_scale:
            metric_logger.update(invisible_lr=optimizer.param_groups[2]["lr"])
            assert len(optimizer.param_groups) == 4
            assert optimizer.param_groups[0]["lr"] == optimizer.param_groups[1]["lr"]
            assert optimizer.param_groups[2]["lr"] == optimizer.param_groups[3]["lr"]

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}