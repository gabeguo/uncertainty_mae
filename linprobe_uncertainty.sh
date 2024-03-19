python main_linprobe.py \
    --model vit_base_patch16 \
    --lower_bound_model /home/gabeguo/uncertainty_mae/cifar100_quantile/lower_encoder_mae.pt \
    --point_bound_model /home/gabeguo/uncertainty_mae/cifar100_quantile/median_encoder_mae.pt \
    --upper_bound_model /home/gabeguo/uncertainty_mae/cifar100_quantile/upper_encoder_mae.pt \
    --scale_factor_path /home/gabeguo/uncertainty_mae/cifar100_quantile/interval_width.pt \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_uncertainty \
    --device cuda

