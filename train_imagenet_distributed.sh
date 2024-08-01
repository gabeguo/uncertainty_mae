#!/bin/bash

WORKDIR=$(pwd)
output_path=/burg/zgroup/users/gzg2104/_imagenet_models/07_28_24/initialTry
JOB_DIR=$(pwd)
python $WORKDIR/submitit_pretrain.py \
    --ngpus 4 \
    --nodes 1 \
    --accum_iter 8 \
    --timeout 720 \
    --job_dir $JOB_DIR \
    --partition short \
    --account zgroup \
    --job_name imagenet_try \
    --output mae.out \
    --error mae.err \
    --dataset_name imagenet \
    --data_path /burg/zgroup/users/gzg2104/data/imagenet/train \
    --batch_size 128 \
    --blr 1.5e-4 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 20 \
    --vae \
    --kld_beta 25 \
    --invisible_lr_scale 0.01 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project imagenet_scaledUp \
    --wandb_name initialTryManitou \
    --disable_zero_conv \
    --object_mask \
    --add_default_mask \
    --var 1 \
    --exclude 'm[012]'