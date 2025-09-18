# rqvae/img_datasets/vehicle.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

# Supported image extensions
IMG_EXTS = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')


class Vehicle(Dataset):
    """
    Vehicle Dataset for image reconstruction tasks.

    Loads all JPG/JPEG/PNG images from root/{split} directory recursively.
    This is an unsupervised reconstruction task, so no labels are returned.

    Args:
        root (str): Root directory containing train/val/test splits
        split (str): Dataset split ('train', 'val', or 'test')
        transform (callable, optional): Transform to be applied on images

    Returns:
        Tensor: Transformed image tensor
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Construct path to split directory
        split_dir = os.path.join(root, split)

        # Recursively find all image files
        self.files = []
        for ext in IMG_EXTS:
            pattern = os.path.join(split_dir, '**', f'*{ext}')
            self.files.extend(glob(pattern, recursive=True))

        # Sort files for reproducibility
        self.files.sort()

        # Validate that images were found
        if len(self.files) == 0:
            raise RuntimeError(
                f'No images found in {split_dir}. '
                f'Supported formats: {IMG_EXTS}'
            )

        print(f'Found {len(self.files)} images in {split_dir}')

    def __len__(self):
        """Return the total number of images"""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load and transform an image.

        Args:
            idx (int): Index of the image

        Returns:
            Tensor: Transformed image tensor
        """
        # Get image path
        img_path = self.files[idx]

        # Load image and convert to RGB
        with Image.open(img_path) as img:
            img = img.convert('RGB')

        # Apply transforms if available
        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_image_path(self, idx):
        """
        Get the file path of an image.

        Args:
            idx (int): Index of the image

        Returns:
            str: Path to the image file
        """
        return self.files[idx]