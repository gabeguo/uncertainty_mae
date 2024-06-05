output_path=/local/zemel/gzg2104/_emoji_models/06_05_24_lessReg_decay_0_01_lrScale_0_2
python main_pretrain.py \
    --data_path /home/gzg2104/uncertainty_mae/dataset_generation/columbia_emoji/train \
    --dataset_name emoji \
    --batch_size 256 \
    --blr 1e-3 \
    --accum_iter 1 \
    --output_dir $output_path \
    --log_dir $output_path \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --epochs 2000 \
    --log_freq 200 \
    --vae \
    --kld_beta 20 \
    --invisible_lr_scale 0.2 \
    --mask_ratio 0.75 \
    --partial_vae \
    --dropout_ratio 0 \
    --eps 1e-6 \
    --weight_decay 0.01 \
    --mixed_precision