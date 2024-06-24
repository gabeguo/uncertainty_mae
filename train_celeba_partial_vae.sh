output_path=/local/zemel/gzg2104/_celeba_models/06_24_24/common_encoder/initialTry
python main_pretrain.py \
    --dataset_name celeba \
    --batch_size 384 \
    --blr 1e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 20 \
    --epochs 400 \
    --log_freq 20 \
    --vae \
    --kld_beta 10 \
    --invisible_lr_scale 0.1 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.01 \
    --mixed_precision \
    --same_encoder \
    --wandb_project celeba_pretrain