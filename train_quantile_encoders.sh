python train_quantile_encoders.py \
    --lr 1e-3 \
    --min_lr 1e-7 \
    --pretrained_weights /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --output_dir cifar100_quantile \
    --batch_size 32