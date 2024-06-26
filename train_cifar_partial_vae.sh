output_path=/local/zemel/gzg2104/_cifar_models/06_25_24_vanilla_lr2e-4
python main_pretrain.py \
    --dataset_name cifar \
    --batch_size 384 \
    --blr 2e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 50 \
    --kld_beta 25 \
    --mask_ratio 0.75 \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.025 \
    --mixed_precision \
    --wandb_project cifar_pretrain \
    --disable_zero_conv