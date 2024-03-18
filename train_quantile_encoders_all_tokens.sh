python train_quantile_encoders.py \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --pretrained_weights /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --output_dir cifar100_quantile_allTokens \
    --batch_size 64 \
    --mask_ratio 0.75 \
    --return_all_tokens \
    --init_student_to_pretrained