python calibrate.py \
    --lower_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/lower_bound_checkpoint-0.pt \
    --upper_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/upper_bound_checkpoint-0.pt \
    --point_model_filepath /home/gabeguo/uncertainty_mae/cifar100_quantile/median_bound_checkpoint-0.pt \
    --gt_model_filepath /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --mask_ratio 0.75 \
    --batch_size 256 \
    --max_scale_factor 1 \
    --step_size 0.1