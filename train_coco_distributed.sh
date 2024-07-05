output_path=/burg/zgroup/users/gzg2104/_coco_models/07_03_24/initial_try
JOB_DIR=$(pwd)
submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 2 \
    --ngpus 4 \
    --use_volta32 \
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
    --disable_zero_conv