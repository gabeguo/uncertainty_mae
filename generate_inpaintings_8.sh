python generate_inpaintings.py \
    --uncertainty_weights /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-799.pth \
    --baseline_weights /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --num_iterations 5000 \
    --num_samples 8 \
    --save_dir /local/zemel/gzg2104/outputs/09_20_24/eight_samples_save_mask \
    --max_mask_ratio 1 \
    --min_mask_ratio 0