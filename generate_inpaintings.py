import sys
import os

import requests
import random
import argparse
from tqdm import tqdm

import torch
import numpy as np

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

CATEGORY_NAMES = ResNet50_Weights.DEFAULT.meta["categories"]

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

def find_infill_portion(reconstruction, mask):
    assert len(reconstruction.shape) == 4, f"{reconstruction.shape}"
    assert reconstruction.shape == mask.shape, f"{reconstruction.shape}, {mask.shape}"
    assert reconstruction.shape[3] == 3, f"{reconstruction.shape}"
    assert mask.shape[0] == 1, f"{mask.shape}"

    mask = mask[0] # we only have one
    compressed_mask = torch.sum(mask, dim=2) # get rid of channels
    assert compressed_mask.shape == reconstruction.shape[1:3] # should just be h, w
    rows_filled = torch.sum(compressed_mask, dim=1).nonzero()
    cols_filled = torch.sum(compressed_mask, dim=0).nonzero()

    if rows_filled.numel() == 0 or cols_filled.numel() == 0:
        return mask

    r_min = torch.min(rows_filled)
    r_max = torch.max(rows_filled)
    c_min = torch.min(cols_filled)
    c_max = torch.max(cols_filled)

    orig_shape = reconstruction.shape[1:3]
    reconstruction = reconstruction[0]
    reconstructed_portion = reconstruction[r_min:r_max, c_min:c_max]
    reconstructed_portion = torch.permute(reconstructed_portion, (2, 0, 1))
    assert reconstructed_portion.shape[0] == 3
    reconstructed_portion = torchvision.transforms.functional.resize(reconstructed_portion, orig_shape)
    reconstructed_portion = torch.permute(reconstructed_portion, (1, 2, 0))
    assert reconstructed_portion.shape[2] == 3

    return reconstructed_portion

def classify(img, classifier):
    img = torch.einsum('nhwc->nchw', img)
    img = img.cuda()

    prediction = classifier(img).squeeze(0).softmax(0)

    class_id = prediction.argmax().item()
    score = prediction[class_id].item()

    return class_id, score

def run_one_image(args, img, model, img_idx, classifier, sample_idx=None,
                mask_ratio=0.75, force_mask=None, mean=imagenet_mean, std=imagenet_std,
                add_default_mask=False):

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
    # infilled portion only, resized to square
    im_infill_square = find_infill_portion(y, mask)

    # classify background and inpainting
    bg_class_id, bg_score = classify(img=im_masked, classifier=classifier)
    print(f"\tbackground prediction: {CATEGORY_NAMES[bg_class_id]}; {bg_score:.3f}")

    ip_class_id, ip_score = classify(img=im_infill, classifier=classifier)
    print(f"\tinfill prediction: {CATEGORY_NAMES[ip_class_id]}; {ip_score:.3f}")

    plt.figure(figsize=(24, 5))
    plt.rcParams.update({'font.size': 22})
    plt.subplot(1, 6, 1)
    show_image(x[0], "original", mean=mean, std=std)
    plt.subplot(1, 6, 2)
    show_image(im_masked[0], "masked", mean=mean, std=std)
    plt.subplot(1, 6, 3)
    show_image(y[0], "reconstruction", mean=mean, std=std)
    plt.subplot(1, 6, 4)
    show_image(im_infill[0], "infilled", mean=mean, std=std)
    plt.subplot(1, 6, 5)
    show_image(im_infill_square, "infilled (resized)", mean=mean, std=std)
    plt.subplot(1, 6, 6)
    show_image(im_paste[0], "reconstruction + visible", mean=mean, std=std)

    save_path = os.path.join(args.save_dir, 
            f"{img_idx}_{'v' if sample_idx is None else sample_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

    plt.figure(figsize=(6, 6))
    show_image(im_infill[0], "", mean=mean, std=std)
    padded_inpaint_save_path = os.path.join(get_inpaint_dir(args), 
            f"{img_idx}_{'v' if sample_idx is None else sample_idx}_padded.png")
    plt.tight_layout(pad=0)
    plt.savefig(padded_inpaint_save_path)
    show_image(im_infill_square, "", mean=mean, std=std)
    square_save_path = os.path.join(get_inpaint_dir(args), 
            f"{img_idx}_{'v' if sample_idx is None else sample_idx}_square.png")
    plt.tight_layout(pad=0)
    plt.savefig(square_save_path)

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

def randomize_mask_layout(mask_layout, mask_ratio=0.75):
    all_indices = [(i, j) for i in range(mask_layout.shape[0]) for j in range(mask_layout.shape[1])]
    random.shuffle(all_indices)
    for i, j in all_indices[:int(mask_ratio * len(all_indices))]:
        mask_layout[i, j] = 0
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

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_weights', type=str,
                        default='/local/zemel/gzg2104/_imagenet_models/08_02_24/revertSmallBatch/checkpoint-580.pth')
    parser.add_argument('--baseline_weights', type=str,
                        default='/home/gzg2104/uncertainty_mae/mae_visualize_vit_base.pth')
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='/local/zemel/gzg2104/outputs/08_08_24_cov')
    parser.add_argument('--random_mask', action='store_true')

    args = parser.parse_args()

    return args

def get_inpaint_dir(args):
    return os.path.join(args.save_dir, 'inpaints')

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    inpaint_dir = get_inpaint_dir(args)
    os.makedirs(inpaint_dir, exist_ok=True)

    test_loader = create_test_loader()

    uncertainty_model_mae = prepare_uncertainty_model(args.uncertainty_weights, 'mae_vit_base_patch16', same_encoder=False, 
                                                    disable_zero_conv=True, var=var)
    model_mae = prepare_model(args.baseline_weights, 'mae_vit_base_patch16')
    print('Model loaded.')

    model_mae = model_mae.cuda()
    model_mae.eval()

    uncertainty_model_mae = uncertainty_model_mae.cuda()
    uncertainty_model_mae.eval()

    # Using pretrained weights:
    classifier = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    classifier = classifier.cuda()
    classifier.eval()

    add_default_mask=True
        
    print(model_mae)
    for idx, img_dict in tqdm(enumerate(test_loader)):
        print(f"img: {idx}")
        plt.rcParams['figure.figsize'] = [5, 5]
        img = img_dict['image']

        if idx == 0:
            img = create_checker()

        assert img.shape == (1, 3, 224, 224)
        img = img.cuda()
        img = img.squeeze()

        torch.manual_seed(idx)
        #print(mask_layout.shape)
        if args.random_mask:
            mask_layout = torch.ones(14, 14).to(device=img.device)
            randomize_mask_layout(mask_layout, mask_ratio=0.75)
            mask_layout = mask_layout.reshape(1, 14, 14)
        elif idx == 0:
            mask_layout = torch.ones(14, 14).to(device=img.device)
            mask_layout[4:8, 4:8] = 0
            mask_layout = mask_layout.reshape(1, 14, 14)
        else:
            mask_layout = img_dict['token_mask'].to(device=img.device)

        keep_indices, mask_indices = get_mask_indices(mask_layout)

        ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
        mask_ratio = 1 - keep_indices.shape[1] / ids_shuffle.shape[1]
        for j in range(args.num_samples):
            run_one_image(args, img, uncertainty_model_mae, mask_ratio=mask_ratio, force_mask=(keep_indices, mask_indices),
                            mean=imagenet_mean, std=imagenet_std, add_default_mask=add_default_mask, img_idx=idx, sample_idx=j,
                            classifier=classifier)
        print('\tbaseline')
        run_one_image(args, img, model_mae, mask_ratio=mask_ratio, force_mask=ids_shuffle, img_idx=idx,
                      classifier=classifier)
        plt.clf()
        if idx == args.num_iterations:
            break

if __name__ == "__main__":
    args = create_args()
    main(args)

