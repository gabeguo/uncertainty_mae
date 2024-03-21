python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_finetune_point \
    --batch_size 128 \
    --epochs 100 \
    --device cuda

