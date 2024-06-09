output_path=/local/zemel/gzg2104/_flower_models/06_09_24_initialTry
python main_pretrain.py \
    --dataset_name flowers \
    --batch_size 256 \
    --blr 1e-3 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 50 \
    --vae \
    --kld_beta 20 \
    --invisible_lr_scale 0.1 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.025 \
    --mixed_precision \
    --wandb_project flowers_pretrain