output_path=/local/zemel/gzg2104/_imagenet_models/08_26_24/normPixLoss
python main_pretrain.py \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --batch_size 256 \
    --blr 1.5e-4 \
    --accum_iter 2 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 10 \
    --num_workers 8 \
    --vae \
    --kld_beta 30 \
    --invisible_lr_scale 0.01 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project imagenet_hippo \
    --disable_zero_conv \
    --master_port 12355 \
    --object_mask \
    --add_default_mask \
    --var 1 \
    --norm_pix_loss \
    --resume /local/zemel/gzg2104/_imagenet_models/08_26_24/normPixLoss/checkpoint-170.pth