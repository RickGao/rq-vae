# rqvae/img_datasets/vehicle.py
import os
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset

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

        self.files = []
        for ext in IMG_EXTS:
            pattern = os.path.join(split_dir, '**', f'*{ext}')
            self.files.extend(glob(pattern, recursive=True))

        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f'No images found in {split_dir}')

        print(f'Found {len(self.files)} images in {split_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        with Image.open(img_path) as img:
            # Convert to RGB
            img = img.convert('RGB')

            # Pad to 1280x736 (add 8 pixels top and bottom)
            # Use ImageOps.expand which preserves RGB channels
            # img = ImageOps.expand(img, border=(0, 8, 0, 8), fill=(0, 0, 0))

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_image_path(self, idx):
        return self.files[idx]