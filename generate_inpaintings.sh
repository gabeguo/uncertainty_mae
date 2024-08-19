python generate_inpaintings.py \
    --uncertainty_weights /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-600.pth \
    --baseline_weights /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --num_iterations 500 \
    --num_samples 3 \
    --save_dir /local/zemel/gzg2104/outputs/08_19_24_500Samples \
    --confidence_threshold 0.3 \
    --max_categories_to_annot 50