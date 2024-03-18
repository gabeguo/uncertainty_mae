import torch
import numpy as np
import os
from uncertainty_vit import EncoderViT
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

def loss(args, img, scale_factor, point_estimator, lower_estimator, upper_estimator,
         gt_model):
    assert isinstance(point_estimator, EncoderViT)
    assert isinstance(lower_estimator, EncoderViT)
    assert isinstance(upper_estimator, EncoderViT)
    assert isinstance(gt_model, EncoderViT)

    img = img.cuda()

    point_estimator.cuda().eval()
    lower_estimator.cuda().eval()
    upper_estimator.cuda().eval()
    gt_model.cuda().eval()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            z_gt = gt_model(img).detach()
            mask_noise = gt_model.get_mask_noise(img).detach() # FIX NOISE

            # TODO: mask the data for lower and upper bounds (since they have uncertainty)? 
            # Possibly only need the masking in training quantile regression
            z_point = point_estimator.forward_fixed_mask(img, 
                            mask_ratio=args.mask_ratio, noise=mask_noise).detach()
            z_lower = lower_estimator.forward_fixed_mask(img, 
                            mask_ratio=args.mask_ratio, noise=mask_noise).detach()
            z_upper = upper_estimator.forward_fixed_mask(img,
                            mask_ratio=args.mask_ratio, noise=mask_noise).detach()

            assert len(z_point.shape) == 2
            assert z_point.shape == z_lower.shape == z_upper.shape

            low = z_point - scale_factor * (z_point - z_lower)
            high = z_point + scale_factor * (z_upper - z_point)

            # failures = torch.logical_or(torch.logical_or(z_point < low, z_point > high), 
            #                             low > high).sum().item()
            # failures = (z_point < low).sum().item()

            # Thanks ChatGPT for vectorization!
            # Creating a boolean mask where z_gt is within [low, high]
            success_mask = (z_gt >= low) & (z_gt <= high)
            # failed_components = torch.where(~success_mask)
            # i = failed_components[0][0]
            # j = failed_components[1][0]
            # print(f"[{low[i, j]:.2f}, {high[i, j]:.2f}]: {z_point[i, j]:.2f}")
            # Counting successes by summing over the boolean mask (True is treated as 1, False as 0)
            success_set_count = success_mask.sum().item()
            # Calculating total items
            total_items = z_point.numel()
            # Computing loss
            loss = 1 - success_set_count / total_items

            assert loss >= 0 and loss <= 1
    # print(f'bad bounds: {failures/total_items}')

    # print(loss)

    return loss

def calibrate(args, dataloader, 
              point_estimator, lower_estimator, upper_estimator,
              gt_model
    ):
    n = len(dataloader.dataset)
    scale_factor = args.max_scale_factor
    ucb = -1
    print(f'num batches in dataloader: {len(dataloader)}')
    pbar = tqdm(total=(int(args.max_scale_factor / args.step_size) + 1) * len(dataloader), 
                desc=f'Calibrating @ risk = {args.risk_level}; error = {args.error_rate}')
    # TODO: switch to binary search to be faster
    while ucb <= args.risk_level:
        last_ucb = ucb
        scale_factor = scale_factor - args.step_size
        ucb = np.sqrt(1 / (2 * n) * np.log(1 / args.error_rate))
        for i, (img, label) in enumerate(dataloader):
            l_i = loss(args, img=img, scale_factor=scale_factor, 
                       point_estimator=point_estimator, 
                       lower_estimator=lower_estimator, upper_estimator=upper_estimator,
                       gt_model=gt_model)
            ucb += l_i * label.shape[0] / n # scale loss by percentage of dataset it makes up
            pbar.set_postfix_str(f"scale_factor = {scale_factor}; ucb = {last_ucb:.3f}; loss = {l_i:.3f}")
            pbar.update(1)
        assert ucb >= 0

        #pbar.set_postfix_str(f"scale_factor = {scale_factor}; ucb = {ucb}")
    scale_factor = scale_factor + args.step_size # backtrack (loop ends on overshoot)

    return scale_factor


def create_args():
    parser = argparse.ArgumentParser()
    # calibration parameters
    parser.add_argument('--risk_level', type=float, default=0.1, 
                        help='percent of latent factors that are allowed to fall outside confidence interval')
    parser.add_argument('--error_rate', type=float, default=0.05,
                        help='chance that we sample a dataset that violates risk level after calibration')
    parser.add_argument('--max_scale_factor', type=float, default=5,
                        help='scaling we use on quantile estimates')
    parser.add_argument('--step_size', type=float, default=0.25,
                        help='step size to use for scale factor calibration')
    # paths to models
    parser.add_argument('--lower_model_filepath', type=str, 
                        default='/home/gabeguo/uncertainty_mae/cifar100_quantile/lower_encoder_mae.pt')
    parser.add_argument('--upper_model_filepath', type=str,
                        default='/home/gabeguo/uncertainty_mae/cifar100_quantile/upper_encoder_mae.pt')
    parser.add_argument('--point_model_filepath', type=str,
                        default='/home/gabeguo/uncertainty_mae/cifar100_quantile/middle_encoder_mae.pt')
    parser.add_argument('--gt_model_filepath', type=str,
                        default='/home/gabeguo/uncertainty_mae/cifar100_train/checkpoint-399.pth')
    # mask percentage
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Percent of image to mask in lower + upper bound creation (not used yet)')
    # path to data
    parser.add_argument('--data_path', default='../', type=str,
                        help='dataset path (we use CIFAR)')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # dataloader
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # output path
    parser.add_argument('--output_dir', type=str, default='./cifar100_quantile',
                        help='where to save scale factor')
    # miscellaneous
    parser.add_argument('--seed', default=0, type=int)
    
    args = parser.parse_args()

    return args

def main():
    args = create_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: better transform
    # TODO: use different component than train set
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    point_estimator = EncoderViT(backbone_path=args.point_model_filepath, freeze_backbone=True, return_all_tokens=False)
    upper_estimator = EncoderViT(backbone_path=args.upper_model_filepath, freeze_backbone=True, return_all_tokens=False)
    lower_estimator = EncoderViT(backbone_path=args.lower_model_filepath, freeze_backbone=True, return_all_tokens=False)
    gt_model = EncoderViT(backbone_path=args.gt_model_filepath, freeze_backbone=True, return_all_tokens=False)

    scale_factor = calibrate(args=args, dataloader=data_loader_train, point_estimator=point_estimator,
                             lower_estimator=lower_estimator, upper_estimator=upper_estimator,
                             gt_model=gt_model)
    print(f'scale_factor = {scale_factor} @ risk = {args.risk_level}; error = {args.error_rate}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(torch.tensor(scale_factor), os.path.join(args.output_dir, 'interval_width.pt'))

    return

if __name__ == "__main__":
    main()