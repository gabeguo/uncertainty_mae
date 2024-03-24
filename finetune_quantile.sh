python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_lower/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_finetune_lower \
    --batch_size 128 \
    --epochs 100 \
    --device cuda

python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_upper/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_finetune_upper \
    --batch_size 128 \
    --epochs 100 \
    --device cuda

python main_finetune.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_median/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_finetune_median \
    --batch_size 128 \
    --epochs 100 \
    --device cuda