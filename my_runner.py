import os

trainCommand = '''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_pretrain.py \
    --data_path /proj/vondrick/datasets/ImageNet-ILSVRC2012 \
    --dataset_name imagenet \
    --batch_size 256 \
    --accum_iter 1 \
    --output_dir /local/vondrick/aniv/uncertainty_mae/outputs \
    --log_dir /local/vondrick/aniv/uncertainty_mae/logs \
    --model mae_vit_base_patch16 \
    --lower 0.05 \
    --median 0.5 \
    --upper 0.95
'''

testCommand = '''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /local/vondrick/aniv/uncertainty_mae/outputs/checkpoint-0.pth \
    --data_path /proj/vondrick/datasets/ImageNet-ILSVRC2012 \
    --dataset_name imagenet \
    --output_dir /local/vondrick/aniv/uncertainty_mae/outputs/linprobe \
    --device cuda
'''

os.system(trainCommand)