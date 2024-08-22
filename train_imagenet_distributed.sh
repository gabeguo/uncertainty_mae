#!/bin/bash

WORKDIR=$(pwd)
JOB_DIR=/burg/zgroup/users/gzg2104/imported_hippo
python $WORKDIR/submitit_pretrain.py \
    --ngpus 4 \
    --nodes 1 \
    --accum_iter 2 \
    --timeout 7200 \
    --job_dir $JOB_DIR \
    --partition short \
    --account zgroup \
    --job_name resume_from_hippo \
    --output mae.out \
    --error mae.err \
    --dataset_name imagenet \
    --data_path /burg/zgroup/users/gzg2104/data \
    --batch_size 128 \
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
    --wandb_project imagenet_hippo \
    --wandb_name resume_from_hippo \
    --disable_zero_conv \
    --add_default_mask \
    --var 1 \
    --exclude 'm[012]'