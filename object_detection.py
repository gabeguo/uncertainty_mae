from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import os
import numpy as np

from tqdm import tqdm
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

CATEGORIES = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]

def process_image(args, img_path, model):
    img = read_image(img_path)
    img = img[:3,:,:]

    # Step 2: Initialize the inference transforms
    preprocess = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [CATEGORIES[i] for i in prediction["labels"]]
    label_nums = [i for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
    im = to_pil_image(box.detach())
    # im.show()
    curr_output_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(curr_output_dir, exist_ok=True)
    output_path = os.path.join(curr_output_dir, os.path.basename(img_path))
    im.save(output_path)

    # print(labels)
    # print(label_nums)

    return [value.item() for value in label_nums]

def calc_co_occurrence(args, dir, model):
    co_occurrence = np.zeros((90, 90))
    for img_name in tqdm(os.listdir(dir)):
        img_path = os.path.join(dir, img_name)
        label_nums = process_image(args, img_path=os.path.join(dir, img_name), model=model)
        # print(label_nums)
        for i in label_nums:
            for j in label_nums:
                co_occurrence[i, j] += 1
    return co_occurrence

def save_co_occurrence(args, co_occurrence, name):
    # save np
    output_path_np = os.path.join(args.output_dir, f"{name}.npy")
    with open(output_path_np, 'wb') as fout:
        np.save(fout, co_occurrence)
    
    # save visual
    plt.rcParams['font.size'] = 5
    sns.heatmap(co_occurrence, square=True, annot=False,
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, norm=LogNorm())
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Object', fontsize=1.25 * plt.rcParams['font.size'])
    plt.ylabel('Object', fontsize=1.25 * plt.rcParams['font.size'])
    plt.tight_layout()
    output_path_visual = os.path.join(args.output_dir, f"{name}.pdf")
    plt.savefig(output_path_visual)
    plt.close('all')

    return

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--box_score_thresh', type=float, default=0.7)

    args = parser.parse_args()

    return args

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    gt_dir = os.path.join(args.input_dir, 'gt')
    inpaint_ours_dir = os.path.join(args.input_dir, 'inpaint_ours')
    inpaint_baseline_dir = os.path.join(args.input_dir, 'inpaint_baseline')

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=args.box_score_thresh)
    model.eval()

    co_occurrence_gt = calc_co_occurrence(args, dir=gt_dir, model=model)
    co_occurrence_ours = calc_co_occurrence(args, dir=inpaint_ours_dir, model=model)
    co_occurrence_baseline = calc_co_occurrence(args, dir=inpaint_baseline_dir, model=model)

    save_co_occurrence(args, co_occurrence_gt, 'co_occurrence_gt')
    save_co_occurrence(args, co_occurrence_ours, 'co_occurrence_ours')
    save_co_occurrence(args, co_occurrence_baseline, 'co_occurrence_baseline')
    
    # TODO: compare

    return

if __name__ == "__main__":
    args = create_args()
    main(args)
