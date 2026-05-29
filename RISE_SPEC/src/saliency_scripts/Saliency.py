#!/usr/bin/env python
# coding: utf-8

# # Randomized Image Sampling for Explanations (RISE)


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
# Import model factory and audio datasets
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.saliency_utils import *
from src.core.explanations import RISE

from core.model_factory import create_model, get_preprocessor
from core.audio_datasets import AudioDataset, create_audio_data_loader
from config.model_configs import get_model_config, get_dataset_config, get_model_config_for_dataset

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="RISE saliency maps for audio classification")
    parser.add_argument("--model_type", default="resnet50", choices=["resnet50", "htsat"],
                        help="Model architecture (default: resnet50)")
    parser.add_argument("--dataset", default="urbansound8k", choices=["esc50", "urbansound8k"],
                        help="Dataset to explain (default: urbansound8k)")
    parser.add_argument("--audio_dir", default=None,
                        help="Raw audio directory (overrides dataset default)")
    parser.add_argument("--datadir", default=None,
                        help="Pre-generated spectrogram directory (overrides dataset default)")
    parser.add_argument("--weights_path", default=None,
                        help="Path to model checkpoint (overrides config default)")
    parser.add_argument("--gpu_batch", type=int, default=150,
                        help="Batch size for mask application (default: 150)")
    parser.add_argument("--htsat_waveform_input", action="store_true",
                        help="Force HTSAT to use waveform input instead of spectrogram")
    parser.add_argument("--use_audio", action="store_true",
                        help="Force use of raw audio even for ResNet50")
    return parser.parse_args()

# Add support for uppercase image extensions
import torchvision.datasets.folder as folder
# Extend supported image extensions to include uppercase variants
folder.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
                        '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF', '.TIFF')


# In[2]:


# Configuration
_cli = parse_args()
MODEL_TYPE = _cli.model_type
dataset_name = _cli.dataset
dataset_config = get_dataset_config(dataset_name)
config = get_model_config_for_dataset(MODEL_TYPE, dataset_name)

args = Dummy()
args.workers = 2
args.audio_dir = _cli.audio_dir or dataset_config['default_audio_dir']
args.datadir   = _cli.datadir   or dataset_config['default_spectrogram_dir']
args.range     = None
args.input_size = config['input_size']
args.gpu_batch  = _cli.gpu_batch

# HTSAT always needs audio (waveform or spectrogram pipeline); ResNet50 defaults to audio too
USE_AUDIO = _cli.use_audio or MODEL_TYPE != 'resnet50' or True


# ## Prepare data

# Replace lines 44-50 with this:
import glob
from PIL import Image

if USE_AUDIO:
    # Use raw audio files
    dataset = AudioDataset(args.audio_dir)
    data_loader = create_audio_data_loader(
        dataset, batch_size=1, num_workers=args.workers, 
        shuffle=False, sampler=RangeSampler(args.range) if args.range else None
    )
else:
    # Use pre-generated spectrograms (backward compatibility)
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
    
# Load model using model factory
htsat_spec_mode = (MODEL_TYPE == 'htsat') and not _cli.htsat_waveform_input
weights_path = _cli.weights_path or config.get('weights_path')
model = create_model(
    MODEL_TYPE,
    weights_path=weights_path,
    num_classes=config['num_classes'],
    htsat_spectrogram_mode=htsat_spec_mode,
)

if USE_AUDIO:
    preprocessor = get_preprocessor(MODEL_TYPE, config, htsat_spectrogram_mode=htsat_spec_mode)

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


def explain_all(data_loader, explainer, use_audio=USE_AUDIO):
    # Get all predicted labels first
    target = np.empty(len(data_loader), np.int64)
    filenames = []
    
    if use_audio:
        # Process audio files
        for i, (filepath, _, path) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
            # Preprocess audio to model input
            model_input = preprocessor(filepath[0]).cuda()
            p, c = torch.max(model(model_input), dim=1)
            target[i] = c[0]
            filenames.append(filepath[0])  # Get original filename
    else:
        # Process pre-generated spectrograms
        for i, (img, _, path) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
            p, c = torch.max(model(img.cuda()), dim=1)
            target[i] = c[0]
            filenames.append(path[0])  # Get original filename

    # Get saliency maps for all images/audio in val loader
    explanations = np.empty((len(data_loader), *args.input_size))
    
    if use_audio:
        # Process audio files
        for i, (filepath, _, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
            # Preprocess audio to model input
            model_input = preprocessor(filepath[0]).cuda()
            saliency_maps = explainer(model_input)
            explanations[i] = saliency_maps[target[i]].cpu().numpy()
    else:
        # Process pre-generated spectrograms
        for i, (img, _, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
            saliency_maps = explainer(img.cuda())
            explanations[i] = saliency_maps[target[i]].cpu().numpy()
            
    return explanations, target, filenames

# Generate explanations for all images
explanations, targets, filenames = explain_all(data_loader, explainer, use_audio=USE_AUDIO)

# Save results
import os
import pickle

output_dir = f'results/{MODEL_TYPE}/saliency/{dataset_name}_saliency_maps/'
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
    'model': MODEL_TYPE,
    'use_audio': USE_AUDIO
}

with open(os.path.join(output_dir, 'saliency_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print(f"Saved saliency maps for {len(dataset)} images to {output_dir}/")

# Optional: Save individual files
def save_individual_saliency_maps(data_loader, explanations, filenames, output_dir='saliency_maps', use_audio=USE_AUDIO):
    """Save individual saliency maps with original image/audio names."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (input_data, _, path) in enumerate(tqdm(data_loader, desc='Saving individual maps')):
        if use_audio:
            # Process audio file
            model_input = preprocessor(input_data[0]).cuda()
        else:
            # Use pre-generated spectrogram
            model_input = input_data.cuda()
            
        p, c = torch.max(model(model_input), dim=1)
        p, c = p[0].item(), c[0].item()
        
        # Get original filename and create new name
        original_path = path[0]
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]
        
        # Handle subdirectories by preserving relative path structure
        ref_dir = args.audio_dir if use_audio else args.datadir
        rel_path = os.path.relpath(original_path, ref_dir)
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
        
        # # Save visualization as image with original name
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.axis('off')
        # plt.title(f'{100*p:.2f}% {get_class_name(c)}')
        # if use_audio:
        #     # For audio, show the preprocessed spectrogram
        #     tensor_imshow(model_input[0])
        # else:
        #     tensor_imshow(input_data[0])
        #
        # plt.subplot(122)
        # plt.axis('off')
        # plt.title(get_class_name(c))
        # if use_audio:
        #     tensor_imshow(model_input[0])
        # else:
        #     tensor_imshow(input_data[0])
        # plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        #
        # plt.savefig(os.path.join(output_path, f'{name_without_ext}_saliency.png'),
        #            bbox_inches='tight', dpi=150)
        # plt.close()

# Save individual files with original names
save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, use_audio=USE_AUDIO)
