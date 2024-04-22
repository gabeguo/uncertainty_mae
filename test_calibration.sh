python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_point/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_upper/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_lower/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_median/checkpoint-85.pth

python test_calibration.py \
    --eval_weights /home/gabeguo/uncertainty_mae/cifar100_linprobe_multiHeadDecoder/checkpoint-85.pth