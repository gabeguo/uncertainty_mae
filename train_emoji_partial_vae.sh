python main_pretrain.py \
    --data_path /home/gabe/uncertainty_mae/dataset_generation/columbia_emoji/train \
    --dataset_name emoji \
    --batch_size 32 \
    --accum_iter 1 \
    --output_dir emoji_train_partial_vae_fullDataset_kldBeta10_dropout0_1 \
    --log_dir emoji_train_partial_vae_fullDataset_kldBeta10_dropout0_1 \
    --model mae_vit_large_patch16 \
    --epochs 4000 \
    --log_freq 250 \
    --vae \
    --kld_beta 10 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0