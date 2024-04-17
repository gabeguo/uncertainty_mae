python main_pretrain.py \
    --data_path /home/gabe/uncertainty_mae/dataset_generation/columbia_emoji \
    --dataset_name emoji \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir emoji_train \
    --log_dir emoji_train \
    --model mae_vit_base_patch16 