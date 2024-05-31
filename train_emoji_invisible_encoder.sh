python main_pretrain.py \
    --data_path /home/gzg2104/uncertainty_mae/dataset_generation/columbia_emoji/train \
    --dataset_name emoji \
    --batch_size 128 \
    --blr 1e-4 \
    --accum_iter 1 \
    --output_dir /local/zemel/gzg2104/emoji_train_invisible_encoder_largeMAE \
    --log_dir /local/zemel/gzg2104/emoji_train_invisible_encoder_largeMAE \
    --model mae_vit_large_patch16 \
    --epochs 4000 \
    --log_freq 200 \
    --kld_beta 1 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0