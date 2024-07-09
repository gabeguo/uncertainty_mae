output_path=/local/zemel/gzg2104/_coco_models/07_09_24/vanilla_mae
python main_pretrain.py \
    --dataset_name coco \
    --batch_size 384 \
    --blr 1.5e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 800 \
    --log_freq 40 \
    --kld_beta 0 \
    --mask_ratio 0.75 \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.025 \
    --mixed_precision \
    --wandb_project coco_pretrain \
    --disable_zero_conv