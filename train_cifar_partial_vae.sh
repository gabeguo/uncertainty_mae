output_path=/local/zemel/gzg2104/_cifar_models/06_16_24_sharedEncoder_vaeBlocks2
python main_pretrain.py \
    --dataset_name cifar \
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
    --kld_beta 15 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.025 \
    --mixed_precision \
    --pretrained_weights /home/gzg2104/uncertainty_mae/pretrained_models/mae_visualize_vit_base.pth \
    --frozen_backbone_epochs 800 \
    --same_encoder \
    --num_vae_blocks 2 \
    --wandb_project cifar_pretrain