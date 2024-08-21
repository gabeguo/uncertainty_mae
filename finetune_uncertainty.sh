python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gzg2104/uncertainty_mae/pretrained_models/mae_pretrain_vit_base.pth \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 1000 \
    --output_dir /local/zemel/gzg2104/_baseline_finetune_end_to_end/imageNet_partialImage \
    --batch_size 512 \
    --accum_iter 1 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_imagenet \
    --device cuda \
    --num_workers 8 \
    --master_port 12357 \
    --keep_ratio 0.25