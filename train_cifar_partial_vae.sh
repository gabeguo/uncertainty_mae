output_path=/local/zemel/gzg2104/_cifar_models/06_19_24_partialVAE_retry/finetune_heads/scale_0_1_block_0_25
num_epochs=400
python main_pretrain.py \
    --dataset_name cifar \
    --batch_size 512 \
    --blr 1e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs $num_epochs \
    --log_freq 50 \
    --vae \
    --kld_beta 5 \
    --invisible_lr_scale 0.1 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.025 \
    --mixed_precision \
    --num_vae_blocks 1 \
    --block_mask_prob 0.25 \
    --wandb_project cifar_pretrain \
    --pretrained_weights /home/gzg2104/uncertainty_mae/pretrained_models/mae_visualize_vit_base.pth \
    --frozen_backbone_epochs $num_epochs \