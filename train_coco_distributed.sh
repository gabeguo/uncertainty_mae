#!/bin/bash
#SBATCH --account=zgroup
#SBATCH --job-name=dummy_mae
#SBATCH --output=mae.out
#SBATCH --error=mae.err
#SBATCH --time=1:00
# #SBATCH --partition=zgroup
#SBATCH --gres=gpu:A6000:4

WORKDIR=$(pwd)
output_path=/burg/zgroup/users/gzg2104/_coco_models/07_11_24/initial_try
JOB_DIR=$(pwd)
python $WORKDIR/submitit_pretrain.py \
    --ngpus 3 \
    --nodes 1 \
    --timeout 720 \
    --job_dir $JOB_DIR \
    --partition zgroup \
    --account zgroup \
    --job_name dummy_mae \
    --output mae.out \
    --error mae.err \
    --dataset_name coco \
    --batch_size 128 \
    --blr 1e-3 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 40 \
    --vae \
    --kld_beta 25 \
    --invisible_lr_scale 0.1 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.025 \
    --mixed_precision \
    --wandb_project coco_pretrain \
    --wandb_name initialTrySlurm \
    --disable_zero_conv