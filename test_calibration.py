import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from uncertainty_vit import ConfidenceIntervalViT
import models_vit
from timm.utils import accuracy
import torch.nn.functional as F
from main_linprobe import create_model, set_head
from tqdm import tqdm
import os

# Thanks ChatGPT!
def reliability_diagram(args, softmax_probs, gt_labels):
    # checking that we have data in expected format
    N = softmax_probs.shape[0]
    assert softmax_probs.shape == (N, args.nb_classes)
    assert gt_labels.shape == (N,)
    assert np.max(gt_labels) < args.nb_classes and np.min(gt_labels) >= 0

    # Get top class predictions and their probabilities
    top_class_predictions = np.argmax(softmax_probs, axis=1)
    top_class_probs = np.max(softmax_probs, axis=1)
    assert top_class_predictions.shape == (N,)
    assert top_class_probs.shape == (N,)

    # Bin samples into k=10 evenly spaced bins based on the top class probability
    bins = np.linspace(0, 1, args.num_bins + 1)
    #print('bins', bins)
    digitized = np.digitize(top_class_probs, bins) - 1 # Bin indices

    # print('top class probs', top_class_probs)
    # print('digitized', digitized)

    # Calculate accuracy for each bin
    bin_accuracy = np.zeros(args.num_bins)
    bin_count = np.zeros(args.num_bins)
    bin_confidence = np.zeros(args.num_bins)
    for i in range(args.num_bins):
        in_bin = digitized == i
        #print('in bin', in_bin, sum(in_bin))
        correct_predictions = top_class_predictions[in_bin] == gt_labels[in_bin]
        #print('correct pred', correct_predictions, sum(correct_predictions))
        for item in correct_predictions:
            assert item in in_bin
        bin_accuracy[i] = sum(correct_predictions) / sum(in_bin) if sum(in_bin) > 0 else np.nan
        bin_count[i] = sum(in_bin)
        bin_confidence[i] = np.mean(top_class_probs[in_bin]) if sum(in_bin) > 0 else np.nan

    ece = np.sum([bin_count[i] / N * np.abs(bin_accuracy[i] - bin_confidence[i])
        for i in range(args.num_bins) if bin_count[i] > 0
    ])
    print(f'ece = {ece:.3f}')

    # Plot reliability diagram
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.xticks(bins, rotation=70)
    plt.yticks(bins)
    plt.grid()
    plt.stairs(values=bin_accuracy, edges=bins, fill=True)
    plt.plot([0, 1], [0, 1])
    plt.plot(bin_confidence, bin_accuracy, label='confidence v. accuracy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Reliability Diagram: ECE = {ece:.3f}')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Plot sample count in each bin
    plt.subplot(1, 2, 2)
    plt.xticks(bins, rotation=70)
    plt.grid()
    plt.stairs(values=bin_count, edges=bins, fill=True)
    plt.title('Samples per Bin')
    plt.xlabel('Confidence')
    plt.ylabel('Sample Count')
    plt.tight_layout()
    #plt.show()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f"{args.output_dir}/calibration_{'ci' if args.use_ci else 'vanilla'}.png")
    plt.close()

    return

@torch.no_grad()
def eval(args):
    if args.use_ci:
        lower_model = create_model(args)
        middle_model = create_model(args)
        upper_model = create_model(args)
        set_head(lower_model, 'cuda')
        set_head(middle_model, 'cuda')
        set_head(upper_model, 'cuda')
        scale_factor = torch.load(args.scale_factor_path, map_location='cuda')
        model = ConfidenceIntervalViT(lower_model=lower_model, middle_model=middle_model, 
                                      upper_model=upper_model, interval_scale=scale_factor)
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
        )
        set_head(model, 'cuda')
    model.load_state_dict(torch.load(args.eval_weights)['model'])
    model.cuda()
    model.eval()

    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_val = datasets.CIFAR100('../data', train=False, download=True, transform=transform_val)
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    all_softmax = list()
    all_labels = list()

    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for idx, (img, label) in pbar:
            img = img.cuda()
            label = label.cuda()

            with torch.cuda.amp.autocast():
                output = model(img)        

                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                pbar.set_postfix_str(f'acc1 = {acc1:.3f}; acc5 = {acc5:.3f}')

                curr_softmax = F.softmax(output, dim=1)
                assert curr_softmax.shape == output.shape
                assert torch.isclose(torch.sum(curr_softmax[0]), torch.tensor(1.0))

                assert label.shape == (img.shape[0],)

                all_softmax.append(curr_softmax)
                all_labels.append(label)
    
        all_softmax = torch.cat(all_softmax, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_softmax.shape == (all_labels.shape[0], args.nb_classes)
        assert all_labels.shape == (len(dataset_val),)
        
        all_softmax = all_softmax.detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()
    
    return all_softmax, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_classes', type=int, default=100)
    parser.add_argument('--num_bins', type=int, default=20)
    parser.add_argument('--eval_weights', type=str, default='/home/gabeguo/uncertainty_mae/cifar100_linprobe_uncertainty_revised/checkpoint-80.pth')
    parser.add_argument('--use_ci', action='store_true')
    parser.add_argument('--model', default='vit_base_patch16', type=str, help='Name of model to test')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--scale_factor_path', default='/home/gabeguo/uncertainty_mae/cifar100_quantile/interval_width.pt', type=str,
                        help='path to scale factor')
    parser.add_argument('--output_dir', default='calibration_vis', type=str)
    args = parser.parse_args()

    # num_samples = 1000
    # softmax_probs = np.full((num_samples, args.nb_classes), 0.1)
    # for i in range(num_samples):
    #     softmax_probs[i, i % args.nb_classes] = i + 1
    # softmax_probs = softmax_probs / np.sum(softmax_probs, axis=1, keepdims=True)
    # gt_labels = np.zeros(num_samples, dtype=np.int_)
    # for i in range(0, num_samples, 2):
    #     gt_labels[i] = i % args.nb_classes
    # #print('gt labels', gt_labels)
    # #gt_labels = np.random.randint(0, args.nb_classes, size=num_samples)
    softmax_probs, gt_labels = eval(args)

    reliability_diagram(args=args, softmax_probs=softmax_probs, gt_labels=gt_labels)