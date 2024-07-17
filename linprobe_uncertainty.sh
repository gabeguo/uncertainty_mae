python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth \
    --dataset_name cifar \
    --data_path /local/zemel/gzg2104/datasets/imagenet \
    --nb_classes 100 \
    --output_dir /local/zemel/gzg2104/_baseline_linprobe/cifar_07_17_24 \
    --batch_size 1024 \
    --log_dir /local/zemel/gzg2104/logs \
    --wandb_project linprobe_cifar \
    --device cuda \
    --master_port 12358

