import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image
from typing import Optional
import librosa

# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

# Custom function to read tensor that handles grayscale images
def read_tensor_rgb(image_path):
    """Read image and convert to RGB tensor with proper normalization."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    img_tensor = preprocess(img)
    return torch.unsqueeze(img_tensor, 0)  # Add batch dimension


# Plots image from tensor
# def tensor_imshow(inp, title=None, **kwargs):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     # Mean and std for ImageNet
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp, **kwargs)
#     if title is not None:
#         plt.title(title)
def tensor_imshow(inp: torch.Tensor, title: Optional[str] = None, **kwargs):
    """Imshow for Tensor (expects tensor **before** normalization)."""
    inp = inp.numpy().transpose((1, 2, 0))  # C×H×W → H×W×C
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # denormalise
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp.squeeze(), cmap="viridis", **kwargs)
    if title is not None:
        plt.title(title)

# Given label number returns class name
# def get_class_name(c):
#     labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
#     return ' '.join(labels[c].split(',')[0].split()[1:])
def get_class_name(c: int) -> str:
    """Returns ESC-50 category string given integer target."""
    # lazily load mapping the first time the function is called
    if not hasattr(get_class_name, "_mapping"):
        # Build mapping one-off
        esc50_csv = os.path.join(os.path.dirname(__file__), "esc50.csv")
        df = pd.read_csv(esc50_csv)
        mapping = df.drop_duplicates("target").set_index("target")["category"].to_dict()
        get_class_name._mapping = mapping
    return get_class_name._mapping[int(c)]

# Image preprocessing function
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
