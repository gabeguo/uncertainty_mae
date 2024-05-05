import emoji
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from tqdm import tqdm
import random

def plot_emoji(emoji_name, output_folder, width, height):
    the_emoji = emoji.emojize(emoji_name, language='alias')
    with Image.new("RGB", (width, height), "#ffffff") as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", size=109)
        _, _, w, h = draw.textbbox((0, 0), the_emoji, font=font)
        draw.text((width/2 - w/2, height/2 - h/2), the_emoji, font=font, embedded_color=True)
        image.save(os.path.join(output_folder, f"{emoji_name[1:-1]}.png"))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='columbia_emoji', type=str)
    parser.add_argument('--train_percent', default=0.9, type=float)
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # print(list(emoji.EMOJI_DATA)[0:5])
    random.seed(42)
    all_emojis = [emoji.demojize(x) for x in list(emoji.EMOJI_DATA)]
    random.shuffle(all_emojis)
    first_val_idx = int(len(all_emojis) * args.train_percent)
    for emoji_name in tqdm(all_emojis[:first_val_idx]):
        plot_emoji(emoji_name=emoji_name, 
                output_folder=train_dir,
                width=224, height=224)
    for emoji_name in tqdm(all_emojis[first_val_idx:]):
        plot_emoji(emoji_name=emoji_name, 
                output_folder=val_dir,
                width=224, height=224)        