python main_pretrain.py \
    --data_path /home/gabeguo/data \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir cifar100_train_median \
    --log_dir cifar100_train_median \
    --model mae_vit_base_patch16 \
    --quantile 0.5

python main_pretrain.py \
    --data_path /home/gabeguo/data \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir cifar100_train_lower \
    --log_dir cifar100_train_lower \
    --model mae_vit_base_patch16 \
    --quantile 0.05

python main_pretrain.py \
    --data_path /home/gabeguo/data \
    --batch_size 64 \
    --accum_iter 1 \
    --output_dir cifar100_train_upper \
    --log_dir cifar100_train_upper \
    --model mae_vit_base_patch16 \
    --quantile 0.95