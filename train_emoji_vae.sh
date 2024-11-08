python main_pretrain.py \
    --data_path /home/gabe/uncertainty_mae/dataset_generation/columbia_emoji_full/train \
    --dataset_name emoji \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir emoji_train_beta5_mask_0_75_keepLastBatch \
    --log_dir emoji_train_beta5_mask_0_75_keepLastBatch \
    --model mae_vit_base_patch16 \
    --epochs 20000 \
    --log_freq 2000 \
    --include_keywords kiss_ couple_with_heart_ people_holding_hands_ \
    --include_any \
    --vae \
    --kld_beta 5 \
    --mask_ratio 0.75