python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /local/zemel/gzg2104/_coco_models/07_03_24/beta25_blr1_5e-4/checkpoint-720.pth \
    --dataset_name imagenet \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 1000 \
    --output_dir /local/zemel/gzg2104/_coco_models/07_03_24/beta25_blr1_5e-4/finetune \
    --batch_size 4096 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_imagenet \
    --device cuda \
    --master_port 12356

