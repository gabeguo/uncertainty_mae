python main_pretrain.py \
    --dataset_name cifar \
    --batch_size 32 \
    --accum_iter 1 \
    --output_dir cifar_train_partial_vae_beta10_dropout0_1 \
    --log_dir cifar_train_partial_vae_beta10_dropout0_1 \
    --model mae_vit_base_patch16 \
    --epochs 600 \
    --log_freq 40 \
    --vae \
    --kld_beta 10 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0.1