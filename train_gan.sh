output_path=/local/zemel/gzg2104/_imagenet_models/11_07_24/occasional_gan

echo "train gan!"

CUDA_VISIBLE_DEVICES=7 python main_pretrain.py \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --batch_size 256 \
    --blr 1.5e-5 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 3 \
    --epochs 15 \
    --log_freq 1 \
    --num_workers 8 \
    --vae \
    --kld_beta 30 \
    --invisible_lr_scale 0.01 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0.5 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project imagenet_hippo \
    --disable_zero_conv \
    --master_port 12357 \
    --object_mask \
    --add_default_mask \
    --var 1 \
    --resume /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-799.pth \
    --gan --gan_lambda 0.01
