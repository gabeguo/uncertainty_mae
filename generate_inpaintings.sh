num_samples=4
img_dir=/local/zemel/gzg2104/outputs/08_26_24/inpaintings

python generate_inpaintings.py \
    --uncertainty_weights /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-799.pth \
    --baseline_weights /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --num_iterations 5000 \
    --num_samples $num_samples \
    --save_dir $img_dir \
    --max_mask_ratio 0.6 \
    --min_mask_ratio 0.2

python object_detection.py \
    --input_dir $img_dir \
    --output_dir /local/zemel/gzg2104/outputs/08_26_24/detection_mAP \
    --box_score_thresh 0.4 \
    --num_trials_inpaint $num_samples