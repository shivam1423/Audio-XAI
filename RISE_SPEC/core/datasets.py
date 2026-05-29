#!/usr/bin/env python
# coding: utf-8

import os
import glob
import torch
import torch.utils.data
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import IMG_EXTENSIONS

# Import new audio datasets for convenience
from core.audio_datasets import AudioDataset, WaveformDataset, create_audio_data_loader


class CustomImageDataset(torch.utils.data.Dataset):
    """Custom dataset for loading images from directory."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        print(f"Searching in: {os.path.abspath(root_dir)}")
        # Find all image files recursively
        for ext in IMG_EXTENSIONS:
            self.image_files.extend(glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True))
        print(self.root_dir)
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0, img_path


def create_data_loader(dataset, args, range_sampler=None):
    """Create data loader for the dataset."""
    if range_sampler is not None:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=range_sampler
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    
    return data_loader
