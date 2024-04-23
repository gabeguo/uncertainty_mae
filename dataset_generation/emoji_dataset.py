import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class EmojiDataset(Dataset):
    def __init__(self, emoji_dir, keywords=None):
        self.emoji_dir = emoji_dir
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize([238, 234, 231], [50, 53, 62])
        ])
        self.keywords = keywords
        self.create_data()

        return
    
    def create_data(self):
        self.filenames = list()
        self.images = list()
        for filename in tqdm(os.listdir(self.emoji_dir)):
            if (self.keywords is not None) \
                and not any(curr_keyword in filename for curr_keyword in self.keywords):
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