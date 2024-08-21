python generate_inpaintings.py \
    --uncertainty_weights /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-600.pth \
    --baseline_weights /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --num_iterations 5000 \
    --num_samples 2 \
    --save_dir /local/zemel/gzg2104/outputs/08_21_24/inpaintings \
    --max_mask_ratio 0.6 \
    --min_mask_ratio 0.2