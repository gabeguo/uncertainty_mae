output_path=/local/zemel/gzg2104/_coco_models/07_22_24/RETRY_beta30_addDefaultMask_prior0_01
python main_pretrain.py \
    --dataset_name coco \
    --batch_size 384 \
    --blr 1.5e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 400 \
    --log_freq 25 \
    --vae \
    --kld_beta 30 \
    --invisible_lr_scale 0.01 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project RETRY_coco_head_to_head \
    --disable_zero_conv \
    --master_port 12355 \
    --object_mask \
    --add_default_mask \
    --var 0.01