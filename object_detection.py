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

import json

from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint

CATEGORIES = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]

def process_image(args, img_path, model):
    img = read_image(img_path)
    img = img[:3,:,:]

    # Step 2: Initialize the inference transforms
    preprocess = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img).cuda()]

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

    return [value.item() for value in label_nums], {key:prediction[key].detach().cpu() for key in prediction}

def get_img_num(img_name):
    the_num = int(img_name.split('_')[0])
    return the_num

def calc_co_occurrence(args, dir, model, num_trials_per_image=1):
    co_occurrence = np.zeros((91, 91))
    img_num_to_prediction = dict()
    for img_name in tqdm(os.listdir(dir)):
        img_path = os.path.join(dir, img_name)
        the_img_num = get_img_num(img_name)
        for _ in range(num_trials_per_image): # sample baseline image multiple times, to be fair
            label_nums, prediction = process_image(args, img_path=os.path.join(dir, img_name), model=model)
            # keep track of the predictions
            if the_img_num not in img_num_to_prediction:
                img_num_to_prediction[the_img_num] = list()
            img_num_to_prediction[the_img_num].append(prediction)
            # keep track of co-occurrence
            for idx_i, label_i in enumerate(label_nums):
                for idx_j, label_j in enumerate(label_nums):
                    if idx_i != idx_j: # don't want object to co-occur with itself
                        co_occurrence[label_i, label_j] += 1
    return co_occurrence, img_num_to_prediction

def get_best_map(gt_detections, pred_detections):
    best_map = 0
    for i in range(len(pred_detections)):
        metric = MeanAveragePrecision(iou_type="bbox") # reset each time
        metric.update(gt_detections, pred_detections[i:i+1])
        assert len(metric.detection_labels) == 1
        best_map = max(best_map, metric.compute()['map'].item())
    return best_map

def get_map(gt, ours, baseline):
    assert len(gt) == len(ours) == len(baseline)
    maps_ours = list()
    maps_baseline = list()
    for img_num in tqdm(gt):
        gt_detections = gt[img_num]
        assert len(gt_detections) == 1
        # multi-trial
        assert len(ours[img_num]) == len(baseline[img_num]), \
            f"{img_num}; ours: {ours[img_num]}; baseline: {baseline[img_num]}"
        # ours
        maps_ours.append(get_best_map(gt_detections, ours[img_num]))
        maps_baseline.append(get_best_map(gt_detections, baseline[img_num]))

    return (np.mean(maps_ours), np.std(maps_ours)), (np.mean(maps_baseline), np.std(maps_baseline))

def save_co_occurrence(args, co_occurrence, name):
    # save np
    output_path_np = os.path.join(args.output_dir, f"{name}.npy")
    with open(output_path_np, 'wb') as fout:
        np.save(fout, co_occurrence)
    
    # save visual
    plt.rcParams['font.size'] = 3.5
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

def save_stats(args, map_ours, map_baseline):
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)
    with open(os.path.join(args.output_dir, 'maps.json'), 'w') as fout:
        json.dump(
            {'map_ours': map_ours, 'map_baseline': map_baseline},
            fout, indent=4
        )

    return

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--box_score_thresh', type=float, default=0.7)
    parser.add_argument('--num_trials_inpaint', type=int, default=2)

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
    model = model.cuda()

    co_occurrence_gt, gt_img_to_pred = calc_co_occurrence(args, dir=gt_dir, 
        model=model, num_trials_per_image=1)
    co_occurrence_ours, ours_img_to_pred = calc_co_occurrence(args, dir=inpaint_ours_dir, 
        model=model, num_trials_per_image=1)
    co_occurrence_baseline, baseline_img_to_pred = calc_co_occurrence(args, dir=inpaint_baseline_dir, 
        model=model, num_trials_per_image=args.num_trials_inpaint)

    map_ours, map_baseline = get_map(gt=gt_img_to_pred, 
        ours=ours_img_to_pred, baseline=baseline_img_to_pred)

    print(f'Our mAP: {map_ours[0]:.3f} +/- {map_ours[1]:.3f}')
    print(f'Baseline mAP: {map_baseline[0]:.3f} +/- {map_baseline[1]:.3f}')
    
    save_co_occurrence(args, co_occurrence_gt, 'co_occurrence_gt')
    save_co_occurrence(args, co_occurrence_ours, 'co_occurrence_ours')
    save_co_occurrence(args, co_occurrence_baseline, 'co_occurrence_baseline')
    save_stats(args, map_ours=map_ours, map_baseline=map_baseline)

    return

if __name__ == "__main__":
    args = create_args()
    main(args)
