python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /local/zemel/gzg2104/_imagenet_models/07_11_24/beta5_blr1e-4_invScale025/checkpoint-400.pth \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 1000 \
    --output_dir /local/zemel/gzg2104/_imagenet_models/07_11_24/beta5_blr1e-4_invScale025/finetune-400 \
    --batch_size 4096 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_imagenet \
    --device cuda \
    --master_port 12356

