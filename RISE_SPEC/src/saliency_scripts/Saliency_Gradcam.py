#!/usr/bin/env python
# coding: utf-8
#  Simple Grad-CAM version of Saliency.py
#  ─────────────────────────────────────

import os, glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.datasets.folder as folder

# Import model factory and audio datasets
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.saliency_utils import *
from core.model_factory import create_model, get_preprocessor, get_target_layer_for_gradcam
from core.audio_datasets import AudioDataset, create_audio_data_loader
from config.model_configs import get_model_config, get_dataset_config, get_model_config_for_dataset

# --- pytorch-grad-cam ------------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# ──────────────────────────────────────
cudnn.benchmark = True

# Accept *.JPEG etc.
folder.IMG_EXTENSIONS += tuple(e.upper() for e in folder.IMG_EXTENSIONS)

# ──────────────────────────────────────
# Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM saliency maps for audio classification")
    parser.add_argument("--model_type", default="resnet50", choices=["resnet50"],
                        help="Model architecture (default: resnet50)")
    parser.add_argument("--dataset", default="urbansound8k", choices=["esc50", "urbansound8k"],
                        help="Dataset to explain (default: urbansound8k)")
    parser.add_argument("--audio_dir", default=None,
                        help="Raw audio directory (overrides dataset default)")
    parser.add_argument("--datadir", default=None,
                        help="Pre-generated spectrogram directory (overrides dataset default)")
    parser.add_argument("--weights_path", default=None,
                        help="Path to model checkpoint (overrides config default)")
    parser.add_argument("--gpu_batch", type=int, default=250,
                        help="Batch size for GradCAM (default: 250)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/saliency/gradcam_{model}_{dataset})")
    return parser.parse_args()

_cli = parse_args()
MODEL_TYPE = _cli.model_type
dataset_config = get_dataset_config(_cli.dataset)
config = get_model_config_for_dataset(MODEL_TYPE, _cli.dataset)

# Args (keep Dummy pattern)
args = Dummy()
args.workers    = 2
args.audio_dir  = _cli.audio_dir or dataset_config['default_audio_dir']
args.datadir    = _cli.datadir   or dataset_config['default_spectrogram_dir']
args.range      = None
args.input_size = config['input_size']
args.gpu_batch  = _cli.gpu_batch
out_dir = _cli.output_dir or f'results/saliency/gradcam_{MODEL_TYPE}_{_cli.dataset}'

# Flag to use audio files (True) or pre-generated spectrograms (False)
USE_AUDIO = True

# ──────────────────────────────────────
# Dataset
if USE_AUDIO:
    dataset = AudioDataset(args.audio_dir)
    data_loader = create_audio_data_loader(dataset, batch_size=1, num_workers=args.workers)
else:
    # Dataset identical to Saliency.py
    class CustomImageDataset(torch.utils.data.Dataset):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
        def __init__(self, root_dir, transform=None):
            self.transform = transform
            self.files = []
            for e in self.exts:
                self.files.extend(glob.glob(os.path.join(root_dir, '**', e), recursive=True))
            print(f'Found {len(self.files)} images')

        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            path  = self.files[idx]
            img   = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, 0, path

    dataset     = CustomImageDataset(args.datadir, preprocess)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

# ──────────────────────────────────────
# Model (keep single-GPU object for grad-cam, no DataParallel/Softmax for GradCAM)
weights_path = _cli.weights_path or config.get('weights_path')
model_wrapped = create_model(MODEL_TYPE, weights_path=weights_path, num_classes=config['num_classes'])

# Extract backbone (unwrap DataParallel and Sequential)
if hasattr(model_wrapped, 'module'):
    backbone = model_wrapped.module[0]  # Unwrap DataParallel, get backbone from Sequential
else:
    backbone = model_wrapped[0]

# Ensure gradients can flow (remove frozen state temporarily for GradCAM)
for p in backbone.parameters():
    p.requires_grad = True

# Get target layers for GradCAM
target_layers = get_target_layer_for_gradcam(backbone, MODEL_TYPE)
gc = GradCAM(model=backbone, target_layers=target_layers, use_cuda=True)

# Get preprocessor if using audio
if USE_AUDIO:
    preprocessor = get_preprocessor(MODEL_TYPE, config)
# ──────────────────────────────────────
# Explanation loop (mirrors Saliency.py)
os.makedirs(out_dir, exist_ok=True)
explanations = np.empty((len(data_loader), *args.input_size))

for i, (input_data, _, path) in enumerate(tqdm(data_loader, desc='Explaining images')):
    if USE_AUDIO:
        # Process audio file
        img_gpu = preprocessor(input_data[0]).cuda().requires_grad_(True)
    else:
        # Use pre-generated spectrogram
        img_gpu = input_data.cuda().requires_grad_(True)

    # get predicted class id
    with torch.no_grad():
        preds = backbone(img_gpu)
        c = preds.argmax(dim=1).item()

    grayscale_cam = gc(input_tensor=img_gpu,
                       targets=[ClassifierOutputTarget(c)])[0]   # (224,224)
    cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max()+1e-8)

    explanations[i] = cam
    name = os.path.splitext(os.path.basename(path[0]))[0]

    # save raw CAM
    np.save(os.path.join(out_dir, f'{name}_gradcam.npy'), cam)

    # Composite overlay PNG
    plt.figure(figsize=(10,5))
    plt.subplot(121); plt.axis('off')
    if USE_AUDIO:
        tensor_imshow(img_gpu[0].detach().cpu())  # Show preprocessed spectrogram
    else:
        tensor_imshow(input_data[0])
    plt.title(get_class_name(c))
    plt.subplot(122); plt.axis('off')
    if USE_AUDIO:
        tensor_imshow(img_gpu[0].detach().cpu())
    else:
        tensor_imshow(input_data[0])
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.savefig(os.path.join(out_dir, f'{name}_gradcam.png'),
                dpi=150,bbox_inches='tight')
    plt.close()

# Save big array for later evaluation (optional)
np.save(os.path.join(out_dir,'all_gradcams.npy'), explanations)
print(f'Done! Saved Grad-CAM maps to {out_dir}/') 