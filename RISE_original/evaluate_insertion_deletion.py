#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.functional import conv2d

from utils import *
from evaluation import CausalMetric, auc, gkern
from explanations import RISE

cudnn.benchmark = True
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images',   required=True)
parser.add_argument('--maps_dir', required=True)
parser.add_argument('--suffix',   default='_saliency.npy')
parser.add_argument('--output_dir', required=True)
args_cli = parser.parse_args()

test_images_dir   = args_cli.images
saliency_maps_dir = args_cli.maps_dir
file_suffix       = args_cli.suffix
output_dir        = args_cli.output_dir

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load model
print("Loading model...")
# Load black box model for explanations
model = models.resnet50(True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
    
# To use multiple GPUs
model = nn.DataParallel(model)

# Define substrate functions
klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

# Initialize metrics
print("Initializing evaluation metrics...")
insertion_metric = CausalMetric(model, 'ins', 224, substrate_fn=blur)
deletion_metric = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)


# Find all image files
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
           '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']:
    image_files.extend(glob.glob(os.path.join(test_images_dir, '**', ext), recursive=True))

print(f"Found {len(image_files)} images")

# Preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Results storage
insertion_scores = []
deletion_scores = []
image_names = []

print("Starting evaluation...")
for img_path in tqdm(image_files, desc="Evaluating images"):
    try:
        # Load and preprocess image
        img_tensor = read_tensor(img_path)
        
        # Get corresponding saliency map
        rel_path = os.path.relpath(img_path, test_images_dir)
        name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        
        # Look for saliency map in the same directory structure
        saliency_path = os.path.join(
        saliency_maps_dir,
        rel_path.replace(os.path.splitext(rel_path)[1], file_suffix)
)
        
        if not os.path.exists(saliency_path):
            # Try without subdirectory
            saliency_path = os.path.join(saliency_maps_dir, f'{name_without_ext}{file_suffix}')
        
        if not os.path.exists(saliency_path):
            print(f"Warning: Saliency map not found for {img_path}")
            continue
            
        # Load saliency map
        saliency_map = np.load(saliency_path)
        
        # Run insertion evaluation
        insertion_score = insertion_metric.single_run(img_tensor, saliency_map, verbose=0)
        insertion_auc = auc(insertion_score)
        
        # Run deletion evaluation
        deletion_score = deletion_metric.single_run(img_tensor, saliency_map, verbose=0)
        deletion_auc = auc(deletion_score)
        
        # Store results
        insertion_scores.append(insertion_auc)
        deletion_scores.append(deletion_auc)
        image_names.append(name_without_ext)
        
        print(f"{name_without_ext}: Insertion AUC = {insertion_auc:.4f}, Deletion AUC = {deletion_auc:.4f}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

# Calculate mean AUCs
if insertion_scores and deletion_scores:
    mean_insertion_auc = np.mean(insertion_scores)
    mean_deletion_auc = np.mean(deletion_scores)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Mean Insertion AUC: {mean_insertion_auc:.4f}")
    print(f"Mean Deletion AUC: {mean_deletion_auc:.4f}")
    print(f"Number of images evaluated: {len(insertion_scores)}")
    
    # Save detailed results
    results = {
        'image_names': image_names,
        'insertion_scores': insertion_scores,
        'deletion_scores': deletion_scores,
        'mean_insertion_auc': mean_insertion_auc,
        'mean_deletion_auc': mean_deletion_auc,
        'total_images': len(insertion_scores)
    }
    
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
    
    # Save as text file for easy reading
    with open(os.path.join(output_dir, 'evaluation_summary_rise.txt'), 'w') as f:
        f.write("=== RISE SALIENCY MAP EVALUATION ===\n\n")
        f.write(f"Mean Insertion AUC: {mean_insertion_auc:.4f}\n")
        f.write(f"Mean Deletion AUC: {mean_deletion_auc:.4f}\n")
        f.write(f"Number of images evaluated: {len(insertion_scores)}\n\n")
        f.write("Detailed Results:\n")
        f.write("Image Name\tInsertion AUC\tDeletion AUC\n")
        f.write("-" * 50 + "\n")
        for name, ins_auc, del_auc in zip(image_names, insertion_scores, deletion_scores):
            f.write(f"{name}\t{ins_auc:.4f}\t{del_auc:.4f}\n")
    
    print(f"\nResults saved to {output_dir}/")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.hist(insertion_scores, bins=10, alpha=0.7, color='blue')
    plt.axvline(mean_insertion_auc, color='red', linestyle='--', label=f'Mean: {mean_insertion_auc:.4f}')
    plt.xlabel('Insertion AUC')
    plt.ylabel('Frequency')
    plt.title('Insertion AUC Distribution')
    plt.legend()
    
    plt.subplot(122)
    plt.hist(deletion_scores, bins=10, alpha=0.7, color='green')
    plt.axvline(mean_deletion_auc, color='red', linestyle='--', label=f'Mean: {mean_deletion_auc:.4f}')
    plt.xlabel('Deletion AUC')
    plt.ylabel('Frequency')
    plt.title('Deletion AUC Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Evaluation completed successfully!")
    
else:
    print("No valid images found for evaluation.")
