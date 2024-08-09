python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --dataset_name cifar \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 100 \
    --output_dir /local/zemel/gzg2104/_baseline_finetune_end_to_end/cifar_08_09_24 \
    --batch_size 512 \
    --accum_iter 2 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_cifar \
    --device cuda \
    --num_workers 8 \
    --master_port 12357
