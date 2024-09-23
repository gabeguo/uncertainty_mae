img_dir=/local/zemel/gzg2104/outputs/09_19_24/four_samples_save_mask

# python generate_inpaintings.py \
#     --uncertainty_weights /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-799.pth \
#     --baseline_weights /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
#     --num_iterations 5000 \
#     --num_samples 4 \
#     --save_dir $img_dir \
#     --max_mask_ratio 1 \
#     --min_mask_ratio 0

# python object_detection.py \
#     --input_dir $img_dir \
#     --output_dir /local/zemel/gzg2104/outputs/09_22_24/objectDetectionMaskedBEiT/four/thresh0_8/multiSampleBEiT \
#     --box_score_thresh 0.8 \
#     --occurrence_prob_threshold 0.1

python object_detection.py \
    --input_dir $img_dir \
    --output_dir /local/zemel/gzg2104/outputs/09_22_24/objectDetectionMaskedBEiT/four/thresh0_6/singleSampleBEiTSoftmax \
    --box_score_thresh 0.6 \
    --occurrence_prob_threshold 0.1 \
    --single_sample_beit --skip_mae

python object_detection.py \
    --input_dir $img_dir \
    --output_dir /local/zemel/gzg2104/outputs/09_22_24/objectDetectionMaskedBEiT/four/thresh0_6/singleSampleBEiTOneHot \
    --box_score_thresh 0.6 \
    --occurrence_prob_threshold 0.1 \
    --single_sample_beit --skip_mae --beit_one_hot