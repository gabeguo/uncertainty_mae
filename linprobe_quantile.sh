python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_lower/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_lower \
    --device cuda

python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_upper/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_upper \
    --device cuda

python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_median/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_median \
    --device cuda

python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /home/gabeguo/uncertainty_mae/cifar100_train_multiDecoder/checkpoint-399.pth \
    --data_path /home/gabeguo/data/cifar-100-python \
    --nb_classes 100 \
    --output_dir cifar100_linprobe_multiHeadDecoder \
    --device cuda
