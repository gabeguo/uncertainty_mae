output_path=/local/zemel/gzg2104/_coco_models/07_24_24/baseline_eps1e-8
python main_pretrain.py \
    --dataset_name coco \
    --batch_size 384 \
    --blr 1.5e-4 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 400 \
    --log_freq 25 \
    --mask_ratio 0.75 \
    --dropout_ratio 0 \
    --eps 1e-8 \
    --weight_decay 0.05 \
    --mixed_precision \
    --wandb_project RETRY_coco_head_to_head \
    --disable_zero_conv \
    --master_port 12354 \
    --object_mask