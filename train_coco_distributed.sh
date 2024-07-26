#!/bin/bash

WORKDIR=$(pwd)
output_path=/burg/zgroup/users/gzg2104/_coco_models/07_25_24/replicateOn_multiNode
JOB_DIR=$(pwd)
python $WORKDIR/submitit_pretrain.py \
    --ngpus 4 \
    --nodes 2 \
    --timeout 720 \
    --job_dir $JOB_DIR \
    --partition short \
    --account zgroup \
    --job_name mae_retry \
    --output mae.out \
    --error mae.err \
    --dataset_name coco \
    --batch_size 128 \
    --blr 1.5e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 400 \
    --log_freq 40 \
    --vae \
    --kld_beta 30 \
    --invisible_lr_scale 0.025 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project RETRY_coco_head_to_head \
    --wandb_name replicate_onMultiNode \
    --disable_zero_conv \
    --object_mask \
    --add_default_mask \
    --var 1