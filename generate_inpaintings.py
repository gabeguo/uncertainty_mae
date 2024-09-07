import sys
import os

import requests
import random
import argparse
from tqdm import tqdm
import json

import torch
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from PIL import Image

import torchvision

import models_mae
from uncertainty_mae import UncertaintyMAE

from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import coco_transforms
from functools import partial

from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
CATEGORIES = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
# ALT_CATEGORIES = [x for x in CATEGORIES[1:] if 'N/A' not in x]

import seaborn as sns

torch.hub.set_dir('/local/zemel/gzg2104/pretrained_weights')

imagenet_mean = 255 * np.array([0.485, 0.456, 0.406])
imagenet_std = 255 * np.array([0.229, 0.224, 0.225])
var = 1

def show_image(image, title='', mean=imagenet_mean, std=imagenet_std):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * std + mean), 0, 255).int())
    if len(title) > 0:
        plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = models_mae.__dict__[arch](norm_pix_loss=False, 
                                    quantile=None, 
                                    vae=False, kld_beta=1)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    print('is vae:', model.vae)
    return model

def prepare_uncertainty_model(chkpt_dir, arch='mae_vit_base_patch16', same_encoder=True, disable_zero_conv=True,
                              var=1):
    visible_model = models_mae.__dict__[arch](norm_pix_loss=False, 
                                    quantile=None, 
                                    vae=False, kld_beta=0)
    invisible_model = models_mae.__dict__[arch](norm_pix_loss=False, 
                                    quantile=None, 
                                    vae=True, kld_beta=0, num_vae_blocks=1, 
                                    disable_zero_conv=disable_zero_conv)
    model = UncertaintyMAE(visible_mae=None if same_encoder else visible_model, 
                           invisible_mae=invisible_model, same_encoder=same_encoder,
                           var=var)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    try:
        msg = model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as the_error:
        print(the_error)
        assert 'invisible_mae.logVar_zero_conv_weight' not in checkpoint
        assert 'invisible_mae.logVar_zero_conv_bias' not in checkpoint
        assert 'invisible_mae.mean_zero_conv_weight' not in checkpoint
        assert 'invisible_mae.mean_zero_conv_bias' not in checkpoint

        msg = model.load_state_dict(checkpoint, strict=False)

        invisible_mae = model.invisible_mae
        invisible_mae.logVar_zero_conv_weight = torch.nn.Parameter(torch.ones(1))
        invisible_mae.logVar_zero_conv_bias = torch.nn.Parameter(torch.zeros(0))
        invisible_mae.mean_zero_conv_weight = torch.nn.Parameter(torch.ones(1))
        invisible_mae.mean_zero_conv_bias = torch.nn.Parameter(torch.zeros(0))

    print(msg)

    return model

def run_one_image(args, img, model, img_idx,
                sample_idx=None, mask_ratio=0.75, force_mask=None, mean=imagenet_mean, std=imagenet_std,
                add_default_mask=True):

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    #x = torch.einsum('nhwc->nchw', x)

    # run MAE
    if isinstance(model, UncertaintyMAE):
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio, force_mask=force_mask, 
                              add_default_mask=add_default_mask, print_stats=False)
        y = model.visible_mae.unpatchify(y)
    else:
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio, force_mask=force_mask, print_stats=False)
        y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    if isinstance(model, UncertaintyMAE):
        mask = mask.unsqueeze(-1).repeat(1, 1, model.visible_mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.visible_mae.unpatchify(mask)  # 1 is removing, 0 is keeping 
    else:
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    # infilled portion, actual size
    im_infill = y * mask

    plt.figure(figsize=(24, 5))
    plt.rcParams.update({'font.size': 22})
    plt.subplot(1, 5, 1)
    show_image(x[0], "original", mean=mean, std=std)
    plt.subplot(1, 5, 2)
    show_image(im_masked[0], "masked", mean=mean, std=std)
    plt.subplot(1, 5, 3)
    show_image(y[0], "reconstruction", mean=mean, std=std)
    plt.subplot(1, 5, 4)
    show_image(im_infill[0], "infilled", mean=mean, std=std)
    plt.subplot(1, 5, 5)
    show_image(im_paste[0], "reconstruction + visible", mean=mean, std=std)

    save_path = os.path.join(get_img_dir(args), 
            f"{img_idx}_{'v' if sample_idx is None else sample_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

    # save full inpainted
    plt.figure(figsize=(6, 6))
    show_image(im_paste[0], "", mean=mean, std=std)
    padded_inpaint_save_path = os.path.join(
        get_inpaint_ours_dir(args) if isinstance(model, UncertaintyMAE) else get_inpaint_baseline_dir(args), 
        f"{img_idx}_{'v' if sample_idx is None else sample_idx}_inpainted.png")
    plt.tight_layout(pad=0)
    plt.savefig(padded_inpaint_save_path)
    # save just inpaint
    show_image(im_infill[0], "", mean=mean, std=std)
    infill_save_path = os.path.join(
        get_infill_ours_dir(args) if isinstance(model, UncertaintyMAE) else get_infill_baseline_dir(args), 
        f"{img_idx}_{'v' if sample_idx is None else sample_idx}_inpainted.png")
    plt.tight_layout(pad=0)
    plt.savefig(infill_save_path)
    # save masked
    show_image(im_masked[0], "", mean=mean, std=std)
    masked_save_path = os.path.join(get_mask_dir(args), f"{img_idx}_mask_image.png")
    plt.tight_layout(pad=0)
    plt.savefig(masked_save_path)
    # save original
    if not isinstance(model, UncertaintyMAE):
        show_image(x[0], "", mean=mean, std=std)
        orig_save_path = os.path.join(get_gt_dir(args), f"{img_idx}_gt_image.png")
        plt.tight_layout(pad=0)
        plt.savefig(orig_save_path)
    plt.close('all')

    return

def create_checker():
    img = torch.zeros(1, 3, 224, 224)
    block_size = 16
    for r in range(14):
        for c in range(14):
            if r % 2 == c % 2:
                val = 1.5
            else:
                val = -1.5
            img[:,:,
                r*block_size:(r+1)*block_size, 
                c*block_size:(c+1)*block_size] = val
    return img

def load_decoder_state_dict(model, chkpt_dir):
    state_dict = torch.load(chkpt_dir)['model']
    # Filter the state_dict to include only the keys for the desired parameters
    filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith((
        'decoder_embed',
        'mask_token',
        'decoder_pos_embed',
        'decoder_blocks',
        'decoder_norm',
        'decoder_pred'
    ))}

    # Load the filtered state_dict into the model
    # Set strict=False to ignore non-matching keys
    model.load_state_dict(filtered_state_dict, strict=False)

    print('loaded decoder')

    return

def create_test_loader():
    ds = load_dataset("detection-datasets/coco")

    ds_val = ds['val']

    custom_transform_function = partial(coco_transforms.transform_function, mask_ratio=None)

    ds_val.set_transform(custom_transform_function)

    test_kwargs = {'batch_size': 1}
    test_loader = torch.utils.data.DataLoader(ds_val, **test_kwargs)

    return test_loader

def randomize_mask_layout(mask_layout):
    rn = random.random()
    assert 0 <= rn <= 1
    if 0 <= rn <= 0.2:
        mask_layout[0:7, 0:7] = 0
    elif 0.2 < rn <= 0.4:
        mask_layout[0:7, 7:14] = 0
    elif 0.4 < rn <= 0.6:
        mask_layout[7:14, 0:7] = 0
    elif 0.6 < rn <= 0.8:
        mask_layout[7:14, 7:14] = 0
    else:
        mask_layout[4:11, 3:10] = 0

    return

def get_mask_indices(mask_layout):
    B = mask_layout.shape[0]
    assert mask_layout.shape == (B, 14, 14), f"{mask_layout.shape}"
    mask_layout = mask_layout.reshape(B, -1)
    keep_indices = torch.where(mask_layout == 1)
    mask_indices = torch.where(mask_layout == 0)
    keep_indices = keep_indices[1].reshape(B, -1)
    mask_indices = mask_indices[1].reshape(B, -1)
    assert keep_indices.shape[0] == B
    assert mask_indices.shape[0] == B
    assert keep_indices.shape[1] + mask_indices.shape[1] == 14 * 14
    assert len(keep_indices.shape) == 2
    assert len(mask_indices.shape) == 2

    return keep_indices, mask_indices

def get_img_dir(args):
    return os.path.join(args.save_dir, 'images')

def get_inpaint_ours_dir(args):
    return os.path.join(args.save_dir, 'inpaint_ours')

def get_inpaint_baseline_dir(args):
    return os.path.join(args.save_dir, 'inpaint_baseline')

def get_gt_dir(args):
    return os.path.join(args.save_dir, 'gt')

def get_infill_ours_dir(args):
    return os.path.join(args.save_dir, 'infillOnly_ours')

def get_infill_baseline_dir(args):
    return os.path.join(args.save_dir, 'infillOnly_baseline')

def get_mask_dir(args):
    return os.path.join(args.save_dir, 'mask')

def get_class_data_dir(args):
    return os.path.join(args.save_dir, 'class_info')

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(get_img_dir(args), exist_ok=True)
    os.makedirs(get_inpaint_ours_dir(args), exist_ok=True)
    os.makedirs(get_inpaint_baseline_dir(args), exist_ok=True)
    os.makedirs(get_infill_ours_dir(args), exist_ok=True)
    os.makedirs(get_infill_baseline_dir(args), exist_ok=True)
    os.makedirs(get_gt_dir(args), exist_ok=True)
    os.makedirs(get_mask_dir(args), exist_ok=True)
    os.makedirs(get_class_data_dir(args), exist_ok=True)

    test_loader = create_test_loader()

    uncertainty_model_mae = prepare_uncertainty_model(args.uncertainty_weights, 'mae_vit_base_patch16', same_encoder=False, 
                                                    disable_zero_conv=True, var=var)
    model_mae = prepare_model(args.baseline_weights, 'mae_vit_base_patch16')
    print('Model loaded.')

    model_mae = model_mae.cuda()
    model_mae.eval()

    uncertainty_model_mae = uncertainty_model_mae.cuda()
    uncertainty_model_mae.eval()

    add_default_mask=True
        
    print(model_mae)
    args.num_iterations = min(args.num_iterations, len(test_loader))
    for idx, img_dict in tqdm(enumerate(test_loader)):
        if idx < args.start_from:
            continue
        print(f"img {idx}:", [CATEGORIES[x] for x in img_dict['masked_classes']])
        print(f"img {idx}:", [CATEGORIES[x] for x in img_dict['classes']])
        if -1 in img_dict['masked_classes']:
            print('SKIP: invalid class!')
            continue
        with open(os.path.join(get_class_data_dir(args), f'{idx}_classes.json'), 'w') as fout:
            json.dump({
                'masked_classes': [x.item() for x in img_dict['masked_classes']],
                'classes': [x.item() for x in img_dict['classes']],
                'masked_classes_str': [CATEGORIES[x] for x in img_dict['masked_classes']],
                'classes_str': [CATEGORIES[x] for x in img_dict['classes']]
            }, fout, indent=4)
        #print(f"img: {idx}")
        plt.rcParams['figure.figsize'] = [5, 5]
        img = img_dict['image']

        assert img.shape == (1, 3, 224, 224)
        img = img.cuda()
        img = img.squeeze()

        torch.manual_seed(idx)
        mask_layout = img_dict['token_mask'].to(device=img.device)
        if args.random_mask \
                or torch.sum(mask_layout) > (1 - args.min_mask_ratio) * 14 * 14 \
                or torch.sum(mask_layout) < (1 - args.max_mask_ratio) * 14 * 14:
            mask_layout = torch.ones(14, 14).to(device=img.device)
            randomize_mask_layout(mask_layout)
            mask_layout = mask_layout.reshape(1, 14, 14)

        keep_indices, mask_indices = get_mask_indices(mask_layout)

        ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
        mask_ratio = 1 - keep_indices.shape[1] / ids_shuffle.shape[1]

        # get the GT only once
        run_one_image(args, img, model_mae, 
                mask_ratio=mask_ratio, force_mask=ids_shuffle, img_idx=idx)
        plt.close('all')

        for j in range(args.num_samples):
            run_one_image(args, img, uncertainty_model_mae, 
                        mask_ratio=mask_ratio, force_mask=(keep_indices, mask_indices),
                        mean=imagenet_mean, std=imagenet_std, add_default_mask=add_default_mask, img_idx=idx, sample_idx=j
            )
            plt.close('all')
        if idx == args.num_iterations:
            break

    with open(os.path.join(args.save_dir, 'settings.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    return

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_weights', type=str,
                        default='/local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-600.pth')
    parser.add_argument('--baseline_weights', type=str,
                        default='/home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth')
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='/local/zemel/gzg2104/outputs/08_08_24_cov')
    parser.add_argument('--random_mask', action='store_true')
    parser.add_argument('--max_mask_ratio', type=float, default=0.7)
    parser.add_argument('--min_mask_ratio', type=float, default=0.2)
    parser.add_argument('--start_from', type=int, default=0)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = create_args()
    main(args)
