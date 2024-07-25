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
        if args.dataset_name in ['imagenet_sketch', 'coco']:
            samples = the_data['image']
        else:
            (samples, _) = the_data
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            if isinstance(model.module, UncertaintyMAE):
                if args.dataset_name == 'coco' and args.object_mask:
                    mask_layout = the_data['token_mask'].to(device=samples.device)
                    B = mask_layout.shape[0]
                    assert mask_layout.shape == (B, 14, 14), f"{mask_layout.shape}"
                    mask_layout = mask_layout.reshape(B, -1)
                    keep_indices = torch.where(mask_layout == 1)
                    mask_indices = torch.where(mask_layout == 0)
                    assert keep_indices[0][0] == keep_indices[0][1] # assert that it's blocked by batch
                    assert mask_indices[0][-1] == mask_indices[0][-2]
                    keep_indices = keep_indices[1].reshape(B, -1) # patches to keep by image
                    mask_indices = mask_indices[1].reshape(B, -1)
                    assert keep_indices.shape[0] == mask_indices.shape[0] == B
                    assert keep_indices.shape[1] + mask_indices.shape[1] == 14 * 14
                    assert len(keep_indices.shape) == 2 and len(mask_indices.shape) == 2
                    ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
                    assert ids_shuffle.shape == (B, 14 * 14)
                    # keep_indices should be B * L
                    mask_ratio = 1 - keep_indices.shape[1] / ids_shuffle.shape[1]
                    force_mask = (keep_indices, mask_indices)
                else:
                    mask_ratio = args.mask_ratio
                    force_mask = None

                loss, pred, mask, reconstruction_loss, kld_loss = \
                    model(samples, mask_ratio=mask_ratio, return_component_losses=True, 
                          force_mask=force_mask, add_default_mask=args.add_default_mask)
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
        if isinstance(model.module, UncertaintyMAE):
            metric_logger.update(reconstruction_loss=reconstruction_loss)
            metric_logger.update(kld_loss=kld_loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if (args.invisible_lr_scale is not None) and (args.invisible_lr_scale > 0) and \
                (not args.same_encoder):
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