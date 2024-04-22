python main_pretrain.py \
    --data_path /home/gabe/uncertainty_mae/dataset_generation/columbia_emoji \
    --dataset_name emoji \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir emoji_train_longer \
    --log_dir emoji_train_longer \
    --model mae_vit_base_patch16 \
    --epochs 8000 \
    --log_freq 1000