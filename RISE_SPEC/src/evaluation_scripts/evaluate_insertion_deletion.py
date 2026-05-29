#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.functional import conv2d
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.saliency_utils import *
from src.evaluation_scripts.evaluation import CausalMetric, auc, gkern
from src.core.explanations import RISE

# Import model factory and preprocessors
from core.model_factory import create_model, get_preprocessor
from config.model_configs import get_model_config, get_dataset_config, get_model_config_for_dataset
from config.settings import AUDIO_EXTENSIONS


cudnn.benchmark = True
from PIL import Image
import torchvision.transforms as transforms
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input',   required=True, help='Directory with test images or audio files')
parser.add_argument('--maps_dir', required=True, help='Directory with saliency maps')
parser.add_argument('--suffix',   default='_saliency.npy', help='Saliency file suffix')
parser.add_argument('--output_dir', required=True, help='Output directory for results')
parser.add_argument('--model_type', type=str, default='resnet50', 
                    choices=['resnet50', 'htsat'], help='Model type')
parser.add_argument('--dataset', type=str, default='esc50',
                    choices=['esc50', 'urbansound8k'],
                    help='Dataset type (default: esc50)')
parser.add_argument('--use_audio', action='store_true', 
                    help='Use raw audio files instead of pre-generated spectrograms')
parser.add_argument('--htsat_waveform_input', action='store_true',
                    help='Force HTSAT to use waveform input (default: spectrogram preprocessing for saliency)')
parser.add_argument('--weights_path', type=str, default=None,
                    help='Path to model checkpoint (overrides config default)')
args_cli = parser.parse_args()

test_images_dir   = args_cli.input
saliency_maps_dir = args_cli.maps_dir
file_suffix       = args_cli.suffix
output_dir        = args_cli.output_dir
MODEL_TYPE        = args_cli.model_type
USE_AUDIO         = args_cli.use_audio

# Get dataset configuration
dataset_config = get_dataset_config(args_cli.dataset)
num_classes = dataset_config['num_classes']

# Get model configuration (dataset-aware: picks correct weights_path and num_classes)
config = get_model_config_for_dataset(MODEL_TYPE, args_cli.dataset)

# Determine if HTSAT should use spectrogram mode (matching saliency generation)
htsat_spectrogram_mode = (MODEL_TYPE == 'htsat') and not args_cli.htsat_waveform_input

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load model using factory with spectrogram mode for HTSAT
weights_path = args_cli.weights_path or config.get('weights_path')
print(f"Loading {MODEL_TYPE} model for {args_cli.dataset} ({num_classes} classes)...")
model = create_model(
    MODEL_TYPE,
    weights_path=weights_path,
    num_classes=num_classes,
    htsat_spectrogram_mode=htsat_spectrogram_mode
)
# Ensure model is in eval mode for stable batch normalization
model.eval()

# Get preprocessor if using audio
if USE_AUDIO:
    preprocessor = get_preprocessor(MODEL_TYPE, config, htsat_spectrogram_mode=htsat_spectrogram_mode)
    if htsat_spectrogram_mode:
        print("Using HTSAT spectrogram preprocessing (matching saliency generation)")
    else:
        print("Using raw audio files with model-specific preprocessing")

# Define substrate functions
klen = 11
ksig = 5

# Create appropriate substrate function based on model type
if MODEL_TYPE == 'htsat':
    # HTSAT: Use mean per frequency bin to preserve frequency structure
    # This matches the "freq" occlusion used in RISE saliency generation
    # This preserves the overall frequency distribution while removing time-specific details
    def htsat_substrate(x):
        """Substrate for HTSAT: use mean per frequency bin to preserve structure.
        
        This matches the baseline computation in RISE when occlusion='freq':
        baseline = x.data.mean(dim=3, keepdim=True).expand_as(x.data)
        """
        # x shape: (B, C, H, W) where H=64 (freq bins), W=1001 (time frames)
        # Compute mean across time dimension (W) for each frequency bin (H)
        # This is the same as RISE's "freq" occlusion baseline
        freq_mean = x.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        # Expand to full time dimension
        return freq_mean.expand_as(x)
    blur = htsat_substrate
    print("HTSAT substrate: Using frequency-preserving baseline (matches RISE 'freq' occlusion)")
else:
    # ResNet: Detect number of channels from config (1 for grayscale, 3 for RGB)
    normalize_mean = config.get('normalize_mean', [0.5, 0.5, 0.5])
    num_channels = len(normalize_mean)
    
    kern = gkern(klen, ksig, num_channels=num_channels).cuda()
    
    # Ensure input is on same device as kernel before convolution
    def blur(x):
        # Move input to same device as kernel if needed
        if x.device != kern.device:
            x = x.to(kern.device)
        return nn.functional.conv2d(x, kern, padding=klen//2)

# Get input size from config (for HTSAT this will be (64, 1001), for ResNet (224, 224))
input_size = config['input_size']  # Full (H, W) tuple
step_size = 224  # Height dimension used as step size

# Initialize metrics with dynamic input size
print("Initializing evaluation metrics...")
print(f"  Input size: {input_size}")
print(f"  Step size: {step_size}")
# Both insertion and deletion should use the same baseline for consistency
# This matches what was used in RISE saliency generation
insertion_metric = CausalMetric(model, 'ins', step_size, substrate_fn=blur, input_size=input_size)
deletion_substrate = lambda x: torch.zeros_like(x)
deletion_metric = CausalMetric(model, 'del', step_size, substrate_fn=deletion_substrate, input_size=input_size)


# Find all files (images or audio)
if USE_AUDIO:
    # Find audio files
    input_files = []
    for ext in AUDIO_EXTENSIONS:
        pattern = f'*{ext}'
        input_files.extend(glob.glob(os.path.join(test_images_dir, '**', pattern), recursive=True))
    print(f"Found {len(input_files)} audio files")
else:
    # Find image files
    input_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
               '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']:
        input_files.extend(glob.glob(os.path.join(test_images_dir, '**', ext), recursive=True))
    print(f"Found {len(input_files)} images")

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

print("Loading all images and saliency maps...")
# Collect all data first for batch processing
img_tensors = []
saliency_maps_list = []
valid_image_names = []

for input_path in tqdm(input_files, desc="Loading"):
    try:
        # Load and preprocess input
        if USE_AUDIO:
            # Process audio file with model-specific preprocessing
            img_tensor = preprocessor(input_path).cuda()
        else:
            # Load pre-generated spectrogram image
            img_tensor = read_tensor_rgb(input_path)
        
        # Get corresponding saliency map
        rel_path = os.path.relpath(input_path, test_images_dir)
        name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
        
        # Look for saliency map in the same directory structure
        saliency_path = os.path.join(
            saliency_maps_dir,
            rel_path.replace(os.path.splitext(rel_path)[1], file_suffix))
        
        if not os.path.exists(saliency_path):
            # Try without subdirectory
            saliency_path = os.path.join(saliency_maps_dir, f'{name_without_ext}{file_suffix}')
        
        if not os.path.exists(saliency_path):
            print(f"Warning: Saliency map not found for {input_path}")
            continue
            
        # Load saliency map
        saliency_map = np.load(saliency_path)
        
        # Handle saliency map shape based on model type
        if saliency_map.ndim == 2:
            # If it's a 2D array (H, W), reshape to (1, H, W)
            saliency_map = saliency_map[np.newaxis, :, :]
        
        if MODEL_TYPE == 'htsat':
            # HTSAT expects 1-channel saliency maps
            if saliency_map.shape[0] == 3:
                # If 3-channel, take the first channel (they should be identical)
                saliency_map = saliency_map[0:1, :, :]
            expected_shape = (1,) + config['input_size']
        else:
            # ResNet: match channels to config (1 for grayscale, 3 for RGB)
            expected_channels = len(config.get('normalize_mean', [0.5, 0.5, 0.5]))
            # if saliency_map.shape[0] == 1 and expected_channels == 3:
            #     # If single channel but model expects RGB, repeat for all 3 channels
            #     saliency_map = np.repeat(saliency_map, 3, axis=0)
            # elif saliency_map.shape[0] == 3 and expected_channels == 1:
            #     # If 3-channel but model expects grayscale, take first channel
            #     saliency_map = saliency_map[0:1, :, :]
            # expected_shape = (expected_channels,) + config['input_size']

            # ResNet/HTSAT: Saliency maps should always be single-channel (H, W)
            if saliency_map.shape[0] == 3:
                # If accidentally saved as 3-channel, average them (they should be identical anyway)
                saliency_map = saliency_map.mean(axis=0, keepdims=True)
            elif saliency_map.ndim == 2:
                # Add channel dimension if 2D
                saliency_map = saliency_map[np.newaxis, :, :]

            expected_shape = (1,) + config['input_size']
        if saliency_map.shape != expected_shape:
            print(f"Warning: Unexpected saliency map shape {saliency_map.shape} for {input_path}, expected {expected_shape}")
            continue
        
        # Store for batch processing (keep on CPU to save GPU memory)
        if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)  # Remove batch dim: (1, C, H, W) -> (C, H, W)
        img_tensors.append(img_tensor.cpu())
        saliency_maps_list.append(saliency_map)
        valid_image_names.append(name_without_ext)
        
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        import traceback
        traceback.print_exc()
        continue

if len(img_tensors) == 0:
    print("No valid images found!")
    exit(1)

print(f"Loaded {len(img_tensors)} images for batch evaluation")

# Stack all images and saliency maps into batches
img_batch = torch.stack(img_tensors)  # Shape: (N, C, H, W)
exp_batch = np.stack(saliency_maps_list)  # Shape: (N, C, H, W)

# Pad to make divisible by batch_size
n_samples = len(img_tensors)

# Determine batch size (adjust based on GPU memory and model type)
if MODEL_TYPE == 'htsat':
    eval_batch_size = min(8, n_samples)  # Smaller batches for HTSAT due to larger input size
else:
    eval_batch_size = min(16, n_samples)  # Larger batches for ResNet

remainder = n_samples % eval_batch_size
if remainder != 0:
    # Pad with last image to make divisible
    pad_count = eval_batch_size - remainder

    img_batch = torch.cat([img_batch, img_batch[-pad_count:]], dim=0)
    exp_batch = np.concatenate([exp_batch, exp_batch[-pad_count:]], axis=0)
    print(f"Padded {pad_count} samples to make batch divisible by {eval_batch_size}")

print(f"Starting batch evaluation: {n_samples} samples, batch_size={eval_batch_size}")
print(f"Total padded samples: {len(img_batch)}")

# Run batch evaluation
print("Running insertion evaluation...")
insertion_scores_batch = insertion_metric.evaluate(img_batch, exp_batch, eval_batch_size)
# insertion_scores_batch shape: (n_steps+1, n_samples_padded)

print("Running deletion evaluation...")
deletion_scores_batch = deletion_metric.evaluate(img_batch, exp_batch, eval_batch_size)
# deletion_scores_batch shape: (n_steps+1, n_samples_padded)

# Extract only the valid samples (remove padding)
insertion_scores_batch = insertion_scores_batch[:, :n_samples]
deletion_scores_batch = deletion_scores_batch[:, :n_samples]

# Compute AUC for each image using the existing auc function
print("Computing AUCs for each image...")
for i in tqdm(range(n_samples), desc="Computing AUCs"):
    insertion_auc = auc(insertion_scores_batch[:, i])
    deletion_auc = auc(deletion_scores_batch[:, i])
    
    insertion_scores.append(insertion_auc)
    deletion_scores.append(deletion_auc)
    image_names.append(valid_image_names[i])
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{n_samples} images...")

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
