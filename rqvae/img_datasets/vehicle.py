# rqvae/img_datasets/vehicle.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch

IMG_EXTS = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')


class Vehicle(Dataset):
    """
    Vehicle Dataset for image reconstruction tasks.
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        split_dir = os.path.join(root, split)

        self.files = glob(os.path.join(split_dir, '**', '*.*'), recursive=True)
        self.files = [f for f in self.files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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

        # Return tuple (image, dummy_label) to match other datasets
        # The trainer expects inputs[0] to be the image
        return (img, 0)  # 返回元组而不是单个图像

    def get_image_path(self, idx):
        return self.files[idx]