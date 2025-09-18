# rqvae/img_datasets/vehicle.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class Vehicle(Dataset):
    """
    Vehicle Dataset for image reconstruction tasks.
    Simple version: loads all images as RGB without padding.
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        split_dir = os.path.join(root, split)

        # Get all files recursively (no extension filtering)
        self.files = glob(os.path.join(split_dir, '**', '*.*'), recursive=True)
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f'No images found in {split_dir}')

        print(f'Found {len(self.files)} images in {split_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        # Load and convert to RGB
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_image_path(self, idx):
        return self.files[idx]