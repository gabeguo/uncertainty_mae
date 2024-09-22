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

def process_image(args, img_path, model, is_beit=False):
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
                            colors="purple",
                            font='/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf',
                            width=4, font_size=45)
    im = to_pil_image(box.detach())
    # im.show()
    curr_output_dir = os.path.join(args.output_dir, f'images{"_beit" if is_beit else ""}')
    os.makedirs(curr_output_dir, exist_ok=True)
    output_path = os.path.join(curr_output_dir, os.path.basename(img_path))
    im.save(output_path)

    # print(labels)
    # print(label_nums)

    return [value.item() for value in label_nums], {key:prediction[key].detach().cpu() for key in prediction}

def get_img_num(img_name):
    the_num = int(img_name.split('_')[0])
    return the_num

def calc_gt_co_occurrence(args, objects_dir):
    # co_occurrence[i, j] = total number of images in which i & j occurred together
    # co_occurrence[i, j] = co_occurrence[j, i]
    # co_occurrence[i, i] = # of total times i appeared (the denominator if we want conditional occurrence prob)
    co_occurrence = np.zeros((91, 91))
    for object_data_file in tqdm(os.listdir(objects_dir)):
        object_data_filepath = os.path.join(objects_dir, object_data_file)
        with open(object_data_filepath, 'r') as fin:
            curr_object_data_dict = json.load(fin)
        # if object occurs multiple times in an image, only count it ONCE
        the_classes = set(curr_object_data_dict["classes"])
        # does this symmetrically
        for i in the_classes:
            for j in the_classes:
                co_occurrence[i, j] += 1
    sanity_check_co_occurrence(co_occurrence)
    return co_occurrence

def sanity_check_co_occurrence(co_occurrence):
    diagonals = set([co_occurrence[i, i] for i in range(co_occurrence.shape[0])])
    assert np.array_equal(co_occurrence.T, co_occurrence)
    assert 0 in diagonals
    return

def get_img_num_to_predLabels(args, inpaint_dir, model, is_beit=False):
    # info we'll need later
    img_num_to_predLabels = dict()
    # go through all the images first
    for img_name in tqdm(os.listdir(inpaint_dir)):
        img_path = os.path.join(inpaint_dir, img_name)
        the_img_num = get_img_num(img_name)
        if is_beit and args.single_sample_beit:
            if "_0_inpainted.png" not in img_path:
                continue
        # if the_img_num > 50:
        #     continue
        # detect what the inpainted object is ONLY
        label_nums, prediction = process_image(args, img_path=img_path, model=model, is_beit=is_beit)
        if the_img_num not in img_num_to_predLabels:
            img_num_to_predLabels[the_img_num] = set()
        # TODO: only have most likely?
        img_num_to_predLabels[the_img_num].update(label_nums)
    return img_num_to_predLabels

def calc_precision_recall(args, inpaint_dir, objects_dir, co_occurrence, model, is_beit=False):
    precisions = list()
    recalls = list()
    precisions_dict = dict()
    recalls_dict = dict()
    precisions_zero_denominator = list()
    recalls_zero_denominator = list()
    img_num_to_predLabels = get_img_num_to_predLabels(args=args,
        inpaint_dir=inpaint_dir, model=model, is_beit=is_beit)
    # now calculate precision and recall per-image
    for the_img_num in img_num_to_predLabels:
        tp = 0
        fp = 0
        fn = 0
        objects_that_should_occur = set(get_objects_that_should_occur(
            args=args, img_num=the_img_num, objects_dir=objects_dir, co_occurrence=co_occurrence))
        assert max(objects_that_should_occur) < 91
        assert len(img_num_to_predLabels[the_img_num]) == 0 \
        or max(img_num_to_predLabels[the_img_num]) < 91
        for class_i in range(co_occurrence.shape[0]):
            if class_i in objects_that_should_occur \
            and class_i in img_num_to_predLabels[the_img_num]:
                tp += 1
            elif class_i not in objects_that_should_occur \
            and class_i in img_num_to_predLabels[the_img_num]:
                fp += 1
            elif class_i in objects_that_should_occur \
            and class_i not in img_num_to_predLabels[the_img_num]:
                fn += 1
        if tp + fp == 0:
            precisions.append(0)
            precisions_zero_denominator.append(the_img_num)
        else:
            precisions.append(tp / (tp + fp))
        precisions_dict[the_img_num] = precisions[-1]
        if tp + fn == 0:
            recalls.append(0)
            recalls_zero_denominator.append(the_img_num)
        else:
            recalls.append(tp / (tp + fn))
        recalls_dict[the_img_num] = recalls[-1]
    return {
        "precisions":precisions, 
        "recalls":recalls, 
        "precisions_dict":precisions_dict, 
        "recalls_dict":recalls_dict,
        "precisions_zero_denominator":precisions_zero_denominator, 
        "recalls_zero_denominator":recalls_zero_denominator
    }

def get_objects_that_should_occur(args, img_num, objects_dir, co_occurrence):
    """
    Returns indices of classes that could appear under the mask,
    conditioned on appearance of other objects (and probability threshold)
    """
    object_data_json_filepath = os.path.join(objects_dir, f"{img_num}_classes.json")
    assert os.path.exists(object_data_json_filepath)
    with open(object_data_json_filepath, 'r') as fin:
        object_data_dict = json.load(fin)
    object_probabilities = noisy_or(object_data_dict['classes'], co_occurrence)
    return set([i for i in range(len(object_probabilities)) \
        if object_probabilities[i] > args.occurrence_prob_threshold])

def noisy_or(class_list, co_occurrence):
    classes = set(class_list)
    probs = list()
    for i in range(co_occurrence.shape[0]):
        # get probability that class i will occur
        total_failure_prob = 1
        for j in classes:
             # number images in which i & j occur together, divided by total appearances of j
            prob_i_given_j = co_occurrence[i, j] / co_occurrence[j, j]
            assert prob_i_given_j <= 1
            curr_failure_prob = 1 - prob_i_given_j
            total_failure_prob *= curr_failure_prob
        prob_i = 1 - total_failure_prob
        probs.append(prob_i)
    return probs


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

def save_stats(args, results):
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)
    with open(os.path.join(args.output_dir, 'precision_recall_by_img.json'), 'w') as fout:
        precision_recall_by_img_dict = dict()
        for method in results:
            precision_recall_by_img_dict[method] = {
                "precision_by_image": results[method]["precisions_dict"],
                "recall_by_image": results[method]["recalls_dict"],
                "precisions_zero_denominator": results[method]["precisions_zero_denominator"],
                "recalls_zero_denominator": results[method]["recalls_zero_denominator"]
            }
        json.dump(precision_recall_by_img_dict, fout, indent=4)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fout:
        aggregate_results = dict()
        for method in results:
            aggregate_results[method] = {
                "precision": {
                    "mean": np.mean(results[method]["precisions"]),
                    "std": np.std(results[method]["precisions"]),
                    "zero_denom": len(results[method]["precisions_zero_denominator"])
                },
                "recall": {
                    "mean": np.mean(results[method]["recalls"]),
                    "std": np.std(results[method]["recalls"]),
                    "zero_denom": len(results[method]["recalls_zero_denominator"])
                }
            }
        json.dump(aggregate_results, fout, indent=4)

    return

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--box_score_thresh', type=float, default=0.7)
    parser.add_argument('--occurrence_prob_threshold', type=float, default=0.05)
    parser.add_argument('--single_sample_beit', action='store_true')
    parser.add_argument('--skip_mae', action='store_true')

    args = parser.parse_args()

    return args

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    objects_dir = os.path.join(args.input_dir, 'class_info')
    # gt_dir = os.path.join(args.input_dir, 'gt')
    inpaint_ours_dir = os.path.join(args.input_dir, 'infillOnly_ours')
    inpaint_baseline_dir = os.path.join(args.input_dir, 'infillOnly_baseline')
    inpaint_beit_dir = os.path.join(args.input_dir, 'infillOnly_beit')

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=args.box_score_thresh)
    model.eval()
    model = model.cuda()

    co_occurrence = calc_gt_co_occurrence(args, objects_dir)
    if args.skip_mae:
        print("Skip MAE!")
    else: # do the MAE
        our_results = \
            calc_precision_recall(args=args, 
            inpaint_dir=inpaint_ours_dir, objects_dir=objects_dir, co_occurrence=co_occurrence, 
            model=model)
        mae_results = \
            calc_precision_recall(args=args, 
            inpaint_dir=inpaint_baseline_dir, objects_dir=objects_dir, co_occurrence=co_occurrence, 
            model=model)
    print("BEiT!")
    beit_results = \
        calc_precision_recall(args=args, 
        inpaint_dir=inpaint_beit_dir, objects_dir=objects_dir, co_occurrence=co_occurrence, 
        model=model, is_beit=True)
    
    save_co_occurrence(args, co_occurrence, 'co_occurrence_gt')
    if args.skip_mae:
        save_stats(args, results={"beit":beit_results})
    else:
        save_stats(args, results={"ours":our_results, "mae":mae_results, "beit":beit_results})

    return

if __name__ == "__main__":
    args = create_args()
    main(args)
