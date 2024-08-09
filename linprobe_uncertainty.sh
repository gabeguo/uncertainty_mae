python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-600.pth \
    --dataset_name cifar \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 100 \
    --output_dir /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/linprobe/600 \
    --batch_size 1024 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_cifar \
    --device cuda \
    --master_port 12358

