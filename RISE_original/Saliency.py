#!/usr/bin/env python
# coding: utf-8

# # Randomized Image Sampling for Explanations (RISE)

# In[1]:


import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *
from explanations import RISE

cudnn.benchmark = True

# Add support for uppercase image extensions
import torchvision.datasets.folder as folder
# Extend supported image extensions to include uppercase variants
folder.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
                        '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF', '.TIFF')


# In[2]:


args = Dummy()

# Number of workers to load data
args.workers = 2
# Directory with images split into class folders.
# Since we don't use ground truth labels for saliency all images can be 
# moved to one class folder.
args.datadir = '/beegfs/data/shared/imagenet/imagenet100/val/'
# Sets the range of images to be explained for dataloader.
args.range = None  # Process all images
# Size of imput images.
args.input_size = (224, 224)
# Size of batches for GPU. 
# Use maximum number that the GPU allows.
args.gpu_batch = 250


# ## Prepare data

# In[3]:

# Custom dataset class to preserve filenames and handle subdirectories
class ImageFolderWithNames(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

# Replace lines 44-50 with this:
import glob
from PIL import Image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Find all image files with various extensions (case-insensitive)
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                   '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']:
            self.image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0, img_path  # Return dummy target and path

dataset = CustomImageDataset(args.datadir, preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
if args.range is not None:
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))
else:
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
# print('Found {: >5} images belonging to {} classes.'.format(len(dataset), len(dataset.classes)))
# print('      {: >5} images will be explained.'.format(len(data_loader) * data_loader.batch_size))


# Load black box model for explanations
model = models.resnet50(True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
    
# To use multiple GPUs
model = nn.DataParallel(model)



explainer = RISE(model, args.input_size, args.gpu_batch)


# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')


# ## Explaining one instance
# Producing saliency maps for top $k$ predicted classes.

# In[7]:


# def example(img, top_k=3):
#     saliency = explainer(img.cuda()).cpu().numpy()
#     p, c = torch.topk(model(img.cuda()), k=top_k)
#     p, c = p[0], c[0]
    
#     plt.figure(figsize=(10, 5*top_k))
#     for k in range(top_k):
#         plt.subplot(top_k, 2, 2*k+1)
#         plt.axis('off')
#         plt.title('{:.2f}% {}'.format(100*p[k], get_class_name(c[k])))
#         tensor_imshow(img[0])

#         plt.subplot(top_k, 2, 2*k+2)
#         plt.axis('off')
#         plt.title(get_class_name(c[k]))
#         tensor_imshow(img[0])
#         sal = saliency[c[k]]
#         plt.imshow(sal, cmap='jet', alpha=0.5)
#         plt.colorbar(fraction=0.046, pad=0.04)

#     plt.show()


# # In[8]:


# example(read_tensor('catdog.png'), 5)


# ## Explaining all images in dataloader
# Explaining the top predicted class for each image.

# In[10]:


def explain_all(data_loader, explainer):
    # Get all predicted labels first
    target = np.empty(len(data_loader), np.int64)
    filenames = []
    for i, (img, _, path) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        p, c = torch.max(model(img.cuda()), dim=1)
        target[i] = c[0]
        filenames.append(path[0])  # Get original filename

    # Get saliency maps for all images in val loader
    explanations = np.empty((len(data_loader), *args.input_size))
    for i, (img, _, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        saliency_maps = explainer(img.cuda())
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    return explanations, target, filenames

# Generate explanations for all images
explanations, targets, filenames = explain_all(data_loader, explainer)

# Save results
import os
import pickle

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

# Save all saliency maps as single numpy file
np.save(os.path.join(output_dir, 'all_saliency_maps.npy'), explanations)

# Save with metadata
results = {
    'explanations': explanations,
    'targets': targets,
    'filenames': filenames,
    'dataset_size': len(dataset),
    'input_size': args.input_size,
    'model': 'resnet50'
}

with open(os.path.join(output_dir, 'saliency_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print(f"Saved saliency maps for {len(dataset)} images to {output_dir}/")

# Optional: Save individual files
def save_individual_saliency_maps(data_loader, explanations, filenames, output_dir='saliency_maps'):
    """Save individual saliency maps with original image names."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img, _, path) in enumerate(tqdm(data_loader, desc='Saving individual maps')):
        p, c = torch.max(model(img.cuda()), dim=1)
        p, c = p[0].item(), c[0].item()
        
        # Get original filename and create new name
        original_path = path[0]
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]
        
        # Handle subdirectories by preserving relative path structure
        rel_path = os.path.relpath(original_path, args.datadir)
        rel_dir = os.path.dirname(rel_path)
        if rel_dir:  # If there's a subdirectory
            subdir_output = os.path.join(output_dir, rel_dir)
            os.makedirs(subdir_output, exist_ok=True)
            output_path = subdir_output
        else:
            output_path = output_dir
        
        # Save saliency map as numpy file with original name
        saliency_map = explanations[i]
        np.save(os.path.join(output_path, f'{name_without_ext}_saliency.npy'), saliency_map)
        
        # Save visualization as image with original name
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.axis('off')
        plt.title(f'{100*p:.2f}% {get_class_name(c)}')
        tensor_imshow(img[0])
        
        plt.subplot(122)
        plt.axis('off')
        plt.title(get_class_name(c))
        tensor_imshow(img[0])
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        
        plt.savefig(os.path.join(output_path, f'{name_without_ext}_saliency.png'), 
                   bbox_inches='tight', dpi=150)
        plt.close()

# Save individual files with original names
save_individual_saliency_maps(data_loader, explanations, filenames, output_dir)

