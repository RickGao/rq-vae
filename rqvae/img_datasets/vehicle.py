# rqvae/img_datasets/vehicle.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch


class Vehicle(Dataset):
    """
    Vehicle Dataset for image reconstruction tasks.
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        split_dir = os.path.join(root, split)

        # Get all image files
        self.files = glob(os.path.join(split_dir, '**', '*.*'), recursive=True)
        # Filter for common image extensions
        self.files = [f for f in self.files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f'No images found in {split_dir}')

        print(f'Found {len(self.files)} images in {split_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        # Load image
        img = Image.open(img_path)
        print(f"[DEBUG] Loaded: mode={img.mode}, size={img.size}, path={os.path.basename(img_path)}")

        # Convert to RGB
        img = img.convert('RGB')
        print(f"[DEBUG] After convert: mode={img.mode}, size={img.size}")

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

            # Check tensor shape after transform
            if isinstance(img, torch.Tensor):
                print(f"[DEBUG] After transform: tensor shape={img.shape}, dtype={img.dtype}")
            else:
                print(f"[DEBUG] After transform: type={type(img)}")

        return img

    def get_image_path(self, idx):
        return self.files[idx]