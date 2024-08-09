python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-600.pth \
    --dataset_name cifar \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 100 \
    --output_dir /local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/finetune_end_to_end/600 \
    --batch_size 512 \
    --accum_iter 2 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_cifar \
    --device cuda \
    --num_workers 16 \
    --master_port 12358
