python main_pretrain.py \
    --data_path /home/gabeguo/data \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir cifar100_train_multiDecoder \
    --log_dir cifar100_train_multiDecoder \
    --model mae_vit_base_patch16 \
    --lower 0.05 \
    --median 0.5 \
    --upper 0.95