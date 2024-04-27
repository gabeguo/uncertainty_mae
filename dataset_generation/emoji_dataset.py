import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class EmojiDataset(Dataset):
    def __init__(self, emoji_dir, include_keywords=None, exclude_keywords=None,
                 include_any=False, exclude_any=False):
        self.emoji_dir = emoji_dir
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize([238, 234, 231], [50, 53, 62])
        ])
        self.include_keywords = include_keywords
        self.exclude_keywords = exclude_keywords
        self.include_any = include_any
        self.exclude_any = exclude_any
        self.create_data()

        return
    
    def create_data(self):
        self.filenames = list()
        self.images = list()
        for filename in tqdm(os.listdir(self.emoji_dir)):
            if (self.include_keywords is not None): # specified keywords to include
                keyword_include_status = [curr_keyword in filename for curr_keyword in self.include_keywords]
                if self.include_any:
                    if (not any(keyword_include_status)): # just needs one desired keyword
                        continue
                elif not all(keyword_include_status): # needs to include all desired keywords to be included
                    continue
            print(filename, 'passed include check')
            if (self.exclude_keywords is not None): # specified keywords to exclude
                keyword_exclude_status = [curr_keyword in filename for curr_keyword in self.exclude_keywords]
                if self.exclude_any:
                    if any(keyword_exclude_status): # can't contain any of the keywords
                        continue
                elif all(keyword_exclude_status): # must contain all to be excluded
                    continue
            filepath = os.path.join(self.emoji_dir, filename)
            assert os.path.exists(filepath)
            self.filenames.append(filepath)
            self.images.append(read_image(filepath).to(dtype=torch.float32))
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), 0