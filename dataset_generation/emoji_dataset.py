import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm

class EmojiDataset(Dataset):
    def __init__(self, emoji_dir, train=True):
        self.emoji_dir = emoji_dir
        self.train_percent = train_percent
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.create_data()

        return
    
    def create_data(self):
        self.filenames = list()
        self.images = list()
        for filename in tqdm(os.listdir(self.emoji_dir)):
            filepath = os.path.join(self.emoji_dir, filename)
            assert os.path.exists(filepath)
            self.filenames.append(filepath)
            self.images.append(read_image(filepath))
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), 0