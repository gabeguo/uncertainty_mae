output_path=/local/zemel/gzg2104/_imagenet_models/07_11_24/beta5_blr1e-4_invScale025
python main_pretrain.py \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --batch_size 384 \
    --blr 1e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 25 \
    --vae \
    --kld_beta 5 \
    --invisible_lr_scale 0.025 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project imagenet_pretrain \
    --disable_zero_conv