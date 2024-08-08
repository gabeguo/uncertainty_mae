from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import random
import torch
import numpy as np

the_pre_transform = transforms.Compose([
    transforms.ToTensor()
])

# Thanks https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
def the_post_transform(image, mask):
    image = TF.to_pil_image(image)
    mask = TF.to_pil_image(mask)
    # Resize
    i_dim = int((1 + 0.1 * random.random()) * 256)
    j_dim = int((1 + 0.1 * random.random()) * 256)
    assert i_dim > 224 and j_dim > 224
    resize = transforms.Resize(size=(i_dim, j_dim))
    image = resize(image)
    mask = resize(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(224, 224))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Transform to tensor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    image = TF.to_tensor(image)
    image = normalize(image)
    mask = TF.to_tensor(mask)
    return image, mask

def transform_function(img_dict, mask_ratio=None):
    img_dict['token_mask'] = [None for _ in range(len(img_dict['image']))]
    for img_idx in range(len(img_dict['image'])):
        img_dict['image'][img_idx] = the_pre_transform(img_dict['image'][img_idx])

        # deal with grayscale
        if img_dict['image'][img_idx].shape[0] == 1:
            img_dict['image'][img_idx] = img_dict['image'][img_idx].repeat(3, 1, 1)

        # pick a bbox
        the_bboxes = img_dict['objects'][img_idx]['bbox']
        if mask_ratio is None:
            height = img_dict['image'][img_idx].shape[1]
            width = img_dict['image'][img_idx].shape[2]
            total_area = height * width
            acceptable_bboxes = list()
            for curr_bbox in the_bboxes:
                x_min, y_min, x_max, y_max = [int(coord) for coord in curr_bbox]
                bbox_area = (x_max - x_min) * (y_max - y_min)
                if 0.3 * total_area < bbox_area < 0.7 * total_area:
                    acceptable_bboxes.append(curr_bbox)
            if len(acceptable_bboxes) == 0:
                x_min = int(0.15 * width)
                x_max = int(0.85 * width)
                y_min = int(0.15 * height)
                y_max = int(0.85 * height)
            else:
                curr_bbox = random.choice(acceptable_bboxes)
                x_min, y_min, x_max, y_max = [int(coord) for coord in curr_bbox]

        # mask in true coordinate space
        fine_mask = torch.ones_like(img_dict['image'][img_idx])
        assert len(fine_mask.shape) == 3, f"{fine_mask.shape}"
        assert fine_mask.shape[0] == 3, f"{fine_mask.shape}"
        fine_mask[:, y_min:y_max, x_min:x_max] = 0
        # do transform only after applying mask
        # transform image and fine mask together
        img_dict['image'][img_idx], fine_mask = the_post_transform(image=img_dict['image'][img_idx], mask=fine_mask)
        
        # set token mask
        fine_mask = fine_mask[0]
        img_dict['token_mask'][img_idx] = create_token_mask(fine_mask, mask_ratio=mask_ratio)

    return {'image':img_dict['image'], 'token_mask':img_dict['token_mask']}

def create_token_mask(fine_mask, dims=(14, 14), mask_ratio=0.75):
    assert len(fine_mask.shape) == 2
    assert fine_mask.shape[0] == fine_mask.shape[1]
    assert len(dims) == 2
    assert dims[0] == dims[1]

    mask_layout = torch.ones(dims)

    for i in range(dims[0]):
        for j in range(dims[1]):
            fine_i_low = int(i / dims[0] * fine_mask.shape[0])
            fine_j_low = int(j / dims[1] * fine_mask.shape[1])
            fine_i_high = int((i + 1) / dims[0] * fine_mask.shape[0])
            fine_j_high = int((j + 1) / dims[1] * fine_mask.shape[1])

            if torch.sum(fine_mask[fine_i_low:fine_i_high, fine_j_low:fine_j_high]) \
                    < (fine_i_high - fine_i_low) * (fine_j_high - fine_j_low):
                mask_layout[i, j] = 0

    if mask_ratio is not None:
        # post-process to make it fit mask ratio
        desired_num_visible = int((1 - mask_ratio) * dims[0] * dims[1])
        curr_num_visible = int(torch.sum(mask_layout).item())
        curr_num_invisible = dims[0] * dims[1] - curr_num_visible
        # too many visible tokens, select some to hide
        if curr_num_visible > desired_num_visible:
            current_visible_tokens = torch.where(mask_layout == 1)
            # print(current_visible_tokens)
            assert len(current_visible_tokens) == 2
            assert current_visible_tokens[0].shape == current_visible_tokens[1].shape
            num_needed_to_mask = curr_num_visible - desired_num_visible
            indices_to_mask = np.random.choice(a=curr_num_visible, size=num_needed_to_mask, replace=False)
            rows_mask = current_visible_tokens[0][indices_to_mask]
            cols_mask = current_visible_tokens[1][indices_to_mask]
            mask_layout[rows_mask, cols_mask] = 0
        # too few visible tokens, select some to reveal
        elif curr_num_visible < desired_num_visible:
            current_invisible_tokens = torch.where(mask_layout == 0)
            assert len(current_invisible_tokens) == 2
            assert current_invisible_tokens[0].shape == current_invisible_tokens[1].shape
            num_needed_to_reveal = desired_num_visible - curr_num_visible
            indices_to_reveal = np.random.choice(a=curr_num_invisible, size=num_needed_to_reveal, replace=False)
            rows_reveal = current_invisible_tokens[0][indices_to_reveal]
            cols_reveal = current_invisible_tokens[1][indices_to_reveal]
            mask_layout[rows_reveal, cols_reveal] = 1
        assert torch.sum(mask_layout) == desired_num_visible

    return mask_layout