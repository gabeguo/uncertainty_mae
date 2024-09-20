import matplotlib.pyplot as plt
import os
import argparse

def plot_images(args, images):
    # show the grid

    f, axarr = plt.subplots(args.n_rows, 3 + args.n_samples, 
        figsize=(3 * (3 + args.n_samples), 3 * args.n_rows))

    plt.rcParams.update({'font.size': 18})

    axarr[0, 0].set_title('Original\n')
    axarr[0, 1].set_title('Masked\n')
    axarr[0, 2].set_title('MAE (Baseline)\n')
    for _c in range(args.n_samples):
        axarr[0, 3 + _c].set_title(f'Partial VAE,\nSample {_c}')
    
    for r in range(args.n_rows):
        axarr[r, 0].imshow(images[r]["orig"])
        for spine in axarr[r, 0].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        axarr[r, 1].imshow(images[r]["masked"])
        for spine in axarr[r, 1].spines.values():
            spine.set_edgecolor('grey')
            spine.set_linewidth(3)
        axarr[r, 2].imshow(images[r]["baseline"])
        for spine in axarr[r, 2].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
        for _c in range(args.n_samples):
            axarr[r, 3+_c].imshow(images[r]["ours"][_c])
            for spine in axarr[r, 3+_c].spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
        for c in range(3 + args.n_samples):
            # axarr[r, c].axis('off')
            axarr[r, c].set_xticks([])
            axarr[r, c].set_yticks([])
        
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
        wspace=0.05, hspace=0.05)

    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'collated.png'))
    plt.savefig(os.path.join(args.output_dir, 'collated.pdf'))

    return

def get_images(args):
    all_image_numbers = [int(filename.split('_')[0]) \
        for filename in os.listdir(os.path.join(args.image_dir, 'class_info'))]
    all_image_numbers.sort()
    desired_img_nums = all_image_numbers[70:70+10*args.n_rows:10] # [103, 118]

    images = list()
    for the_num in desired_img_nums:
        if args.object_detection_dir is None:
            baseline_img = plt.imread(os.path.join(args.image_dir,
                "inpaint_baseline", f"{the_num}_v_inpainted.png"))
            inpaint_ours = [plt.imread(os.path.join(args.image_dir,
                "inpaint_ours", f"{the_num}_{sample_idx}_inpainted.png")) \
                for sample_idx in range(args.n_samples)]
        else:
            baseline_img = plt.imread(os.path.join(args.object_detection_dir,
                "images", f"{the_num}_v_inpainted.png"))
            inpaint_ours = [plt.imread(os.path.join(args.object_detection_dir,
                "images", f"{the_num}_{sample_idx}_inpainted.png")) \
                for sample_idx in range(args.n_samples)]

        images.append({
            "orig": plt.imread(os.path.join(args.image_dir, 
                "gt", f"{the_num}_gt_image.png")),
            "masked": plt.imread(os.path.join(args.image_dir,
                "mask", f"{the_num}_mask_image.png")),
            "baseline": baseline_img,
            "ours": inpaint_ours
        })
    return images

def parse_args():
    parser = argparse.ArgumentParser('plot image grid', add_help=False)
    parser.add_argument('--n_rows', default=6, type=int)
    parser.add_argument('--n_samples', default=4, type=int)
    parser.add_argument('--image_dir', default='/local/zemel/gzg2104/outputs/09_07_24/lessCrop', type=str)
    parser.add_argument('--output_dir', default='/local/zemel/gzg2104/outputs/09_07_24/lessCrop', type=str)
    parser.add_argument('--object_detection_dir', default=None, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    images = get_images(args)
    plot_images(args, images)