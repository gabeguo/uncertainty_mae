python main_pretrain.py \
    --data_path /home/gabeguo/data \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir cifar100_train \
    --log_dir cifar100_train \
    --model mae_vit_base_patch16 