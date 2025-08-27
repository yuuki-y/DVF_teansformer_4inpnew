import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class XrayFolderDataset(Dataset):
    """
    Dataset to load data from folders as specified in the request.
    It expects a root directory with the following structure:
    - root_dir/
      - frontal1/
      - frontal2/
      - lateral1/
      - lateral2/
      - target/
    It assumes that files are sorted alphabetically and correspond to each other.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.view_dirs = {
            'frontal1': os.path.join(root_dir, 'frontal1'),
            'frontal2': os.path.join(root_dir, 'frontal2'),
            'lateral1': os.path.join(root_dir, 'lateral1'),
            'lateral2': os.path.join(root_dir, 'lateral2'),
        }
        self.target_dir = os.path.join(root_dir, 'target')

        # Get the list of files, assuming they are consistently named and sorted
        self.file_list = sorted(os.listdir(self.view_dirs['frontal1']))

        # Basic validation
        for view, path in self.view_dirs.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Directory not found: {path}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Directory not found: {self.target_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # Load the 4 input images
        images = []
        for view in ['frontal1', 'frontal2', 'lateral1', 'lateral2']:
            img_path = os.path.join(self.view_dirs[view], filename)
            # Assuming .pt files, use torch.load
            images.append(torch.load(img_path))

        input_tensor = torch.stack(images) # Stacks to (4, 1, 256, 256)

        # Load the target .nii.gz file
        target_filename = filename.replace('.pt', '.nii.gz')
        target_path = os.path.join(self.target_dir, target_filename)
        nii_img = nib.load(target_path)
        # Convert to numpy array and then to tensor
        # Nibabel loads as (W, H, D, C), we need (C, D, H, W) for PyTorch
        target_array = nii_img.get_fdata(dtype=np.float32)
        target_tensor = torch.from_numpy(target_array).permute(3, 2, 1, 0)

        return input_tensor, target_tensor

class DummyXrayDataset(Dataset):
    """
    A dummy dataset that generates random data samples in memory.
    This is used to demonstrate the model's functionality without requiring
    a real dataset.
    """
    def __init__(self, num_samples=100, img_size=256, out_dims=3, out_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        self.out_dims = out_dims
        self.out_size = out_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: (4, 1, img_size, img_size)
        input_tensor = torch.rand(4, 1, self.img_size, self.img_size)
        # Target: (3, out_size, out_size, out_size)
        target_tensor = torch.rand(self.out_dims, self.out_size, self.out_size, self.out_size)
        return input_tensor, target_tensor
