python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_uncertainty_revised/checkpoint-85.pth \
    --use_ci

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_point/checkpoint-89.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_finetune_point/checkpoint-99.pth \
    --global_pool