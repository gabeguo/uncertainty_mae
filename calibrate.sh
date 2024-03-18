python calibrate.py \
    --lower_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/lower_encoder_mae.pt \
    --upper_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/upper_encoder_mae.pt \
    --point_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/median_encoder_mae.pt \
    --gt_model_filepath /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --mask_ratio 0.75 \
    --batch_size 256 \
    --max_scale_factor 2 \
    --step_size 0.1