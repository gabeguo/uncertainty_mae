python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_lower/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_lower \
    --device cuda

