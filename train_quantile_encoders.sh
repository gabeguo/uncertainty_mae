python train_quantile_encoders.py \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --num_unshared_layers 3 \
    --pretrained_weights /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --output_dir cifar100_quantile