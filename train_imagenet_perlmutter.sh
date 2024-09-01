#!/bin/bash

WORKDIR=$(pwd)
JOB_DIR=$PSCRATCH/weights/09_01_24/normPixLoss
python $WORKDIR/submitit_pretrain.py \
    --ngpus 4 \
    --nodes 8 \
    --accum_iter 1 \
    --timeout 2880 \
    --job_dir $JOB_DIR \
    --partition m1266 \
    --account m1266 \
    --job_name perlmutter \
    --qos regular \
    --output mae.out \
    --error mae.err \
    --dataset_name imagenet \
    --data_path $PSCRATCH/imagenet \
    --batch_size 64 \
    --blr 1.5e-4 \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 10 \
    --vae \
    --kld_beta 30 \
    --invisible_lr_scale 0.01 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project imagenet_perlmutter \
    --wandb_name perlmutter \
    --disable_zero_conv \
    --add_default_mask \
    --var 1 \
    --norm_pix_loss