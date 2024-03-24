python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_point/checkpoint-89.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_finetune_point/checkpoint-99.pth \
    --global_pool

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_upper/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_lower/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_finetune_lower/checkpoint-99.pth \
    --global_pool

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_finetune_upper/checkpoint-99.pth \
    --global_pool