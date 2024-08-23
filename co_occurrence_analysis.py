import numpy as np
import os
from object_detection import save_co_occurrence, CATEGORIES
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/local/zemel/gzg2104/outputs/08_21_24/detection')
    parser.add_argument('--output_dir', type=str, default='dummy')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--kld_smooth', type=float, default=1e-4)

    args = parser.parse_args()

    return args

def get_most_popular_categories(dist_gt, k=10):
    """
    Gets k most popular categories from dist_gt
    """
    top_categories = dist_gt.argsort()[-k:][::-1]
    return top_categories

def overlay_plot(args, dist_gt, dist_ours, dist_baseline, obj_name):
    """
    dist_gt is a co-occurrence distribution for a particular object (based on gt)
    dist_ours and dist_baseline are co-occurrence distribution for that same object (based on in-fills)
    """

    plt.rcParams['font.size'] = 10

    top_categories = get_most_popular_categories(dist_gt, k=args.k)

    print(top_categories)

    ind = np.arange(args.k)
    width = 0.3

    selected_categories = [CATEGORIES[cat].replace(' ', '\n') for cat in top_categories]

    plt.bar(ind, dist_gt[top_categories], 
        width=width, label='GT', color='#aa000088', hatch='+')
    plt.bar(ind - width, dist_ours[top_categories], 
        width=width, label='Partial VAE', color='#0000aa88', hatch='/')
    plt.bar(ind + width, dist_baseline[top_categories],
        width=width, label='Vanilla MAE', color='#00aa0088', hatch='\\')
    plt.title(f'Co-Occurrence for {obj_name.capitalize()}')

    max_val = int(max(
        max(dist_gt[top_categories]), 
        max(dist_ours[top_categories]), 
        max(dist_baseline[top_categories])
    ) * 10 + 1)

    plt.legend()
    plt.xticks(ind, selected_categories, rotation=30, fontsize=7)
    plt.yticks([i / 10 for i in range(max_val)])
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, f"{obj_name}_co-occurrence.pdf"))
    plt.close('all')

    return

def convert_to_distribution(co_occurrence):
    """
    Normalizes the matrix, such that each row sums to 1, like in a probability distribution.
    """
    for i in range(co_occurrence.shape[0]):
        if np.sum(co_occurrence[i]) == 0:
            continue
        co_occurrence[i] /= np.sum(co_occurrence[i])
    
    return co_occurrence

def smooth(args, distribution):
    smoothed = distribution + args.kld_smooth
    smoothed /= np.sum(smoothed)

    return smoothed

def plot_js_distribution(args, gt_results, our_results, baseline_results):
    """
    Calculates the distribution of Jensen-Shannon distance by category between
    gt_results and (our_results, baseline_results)
    """
    our_js = list()
    baseline_js = list()

    for i in range(gt_results.shape[0]):
        if 'n/a' in CATEGORIES[i].lower():
            continue
        smoothed_gt = smooth(args, gt_results[i])
        smoothed_ours = smooth(args, our_results[i])
        smoothed_baseline = smooth(args, baseline_results[i])

        our_js.append(jensenshannon(smoothed_gt, smoothed_ours))
        baseline_js.append(jensenshannon(smoothed_gt, smoothed_baseline))
    
    max_jsd = ((max(max(our_js), max(baseline_js)) // 0.01) + 1) * 0.01

    plt.hist(our_js, bins=np.arange(0, max_jsd, 0.01), color='#0000ccbb', hatch='/', histtype='step', label='Partial VAE')
    plt.hist(baseline_js, bins=np.arange(0, max_jsd, 0.01), color='#00cc00bb', hatch='\\', histtype='step', label='Vanilla MAE')
    plt.xlabel('Jensen-Shannon Divergence Between Co-Occurrence Distributions')
    plt.ylabel('Number of Distributions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'Jensen-Shannon.pdf'))

    print(f'Our JSD: {np.mean(our_js):.3f} +/- {np.std(our_js):.3f}')
    print(f'Their JSD: {np.mean(baseline_js):.3f} +/- {np.std(baseline_js):.3f}')

    return

def main(args):
    gt_results = np.load(os.path.join(args.data_dir, 
        'co_occurrence_gt.npy'))
    baseline_results = np.load(os.path.join(args.data_dir, 
        'co_occurrence_baseline.npy'))
    our_results = np.load(os.path.join(args.data_dir, 
        'co_occurrence_ours.npy'))

    gt_results = convert_to_distribution(gt_results)
    baseline_results = convert_to_distribution(baseline_results)
    our_results = convert_to_distribution(our_results)

    baseline_deviation = baseline_results - gt_results
    save_co_occurrence(args, baseline_deviation, 'gt_vs_baseline')
    print('Baseline:', np.mean(baseline_deviation), np.std(baseline_deviation))

    our_deviation = our_results - gt_results
    save_co_occurrence(args, our_deviation, 'gt_vs_ours')
    print('Ours:', np.mean(our_deviation), np.std(our_deviation))

    desired_objects = ['car', 'person', 'cow', 'banana', 'chair']
    for the_obj in desired_objects:
        obj_idx = CATEGORIES.index(the_obj)
        overlay_plot(args, dist_gt=gt_results[obj_idx], dist_ours=our_results[obj_idx], dist_baseline=baseline_results[obj_idx],
            obj_name=the_obj)

    plot_js_distribution(args, gt_results=gt_results, our_results=our_results, baseline_results=baseline_results)

    return

if __name__ == "__main__":
    args = create_args()
    main(args)