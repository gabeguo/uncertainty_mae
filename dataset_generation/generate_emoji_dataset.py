import emoji
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from tqdm import tqdm

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
    parser.add_argument('--output_dir', default='/home/gabeguo/columbia_emoji', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # print(list(emoji.EMOJI_DATA)[0:5])
    all_emojis = [emoji.demojize(x) for x in list(emoji.EMOJI_DATA)]
    for emoji_name in tqdm(all_emojis):
        plot_emoji(emoji_name=emoji_name, 
                output_folder=args.output_dir,
                width=224, height=224)