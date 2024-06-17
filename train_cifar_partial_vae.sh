output_path=/local/zemel/gzg2104/_cifar_models/06_17_24_fromScratch
python main_pretrain.py \
    --dataset_name cifar \
    --batch_size 512 \
    --blr 1e-3 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 50 \
    --vae \
    --kld_beta 5 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.025 \
    --mixed_precision \
    --same_encoder \
    --num_vae_blocks 1 \
    --wandb_project cifar_pretrain