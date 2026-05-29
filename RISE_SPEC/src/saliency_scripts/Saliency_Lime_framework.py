#!/usr/bin/env python
# coding: utf-8
"""
LIME Framework for HTSAT and ResNet50
---------------------------------
Usage
-----
Run experiments with different model and dataset combinations:

    # ESC50 experiments
    python Saliency_Lime_framework.py --model_type resnet50 --dataset esc50
    python Saliency_Lime_framework.py --model_type htsat --dataset esc50
    
    # UrbanSound8K experiments
    python Saliency_Lime_framework.py --model_type resnet50 --dataset urbansound8k
    python Saliency_Lime_framework.py --model_type htsat --dataset urbansound8k

The results will be stored in ``saliency/saliency_lime_{model_type}_{dataset}/`` next to the script.

You can override default data directories with --datadir and --audio_dir if needed.
"""

import os
import glob
import warnings
import argparse
from typing import Tuple, Union
import sys

import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder

from lime import lime_image
from skimage.segmentation import quickshift

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.saliency_utils import *  # provides preprocess, tensor_imshow, get_class_name, Dummy
from core.model_factory import create_model, get_preprocessor
from core.audio_datasets import AudioDataset, create_audio_data_loader
from config.model_configs import get_model_config, get_dataset_config, get_model_config_for_dataset

# Base directory for the project (parent of RISE_Spec_Waveform_framework)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)  # suppress skimage warnings

# Accept *.JPEG etc. ---------------------------------------------------------
folder.IMG_EXTENSIONS += tuple(e.upper() for e in folder.IMG_EXTENSIONS)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LIME saliency maps for HTSAT and ResNet50")
    
    parser.add_argument("--model_type", type=str, default='resnet50',
                        choices=['resnet50', 'htsat'],
                        help="Model architecture to use (default: resnet50)")
    parser.add_argument("--dataset", type=str, default='urbansound8k',
                        choices=['esc50', 'urbansound8k'],
                        help="Dataset to use (default: urbansound8k)")
    parser.add_argument("--datadir", type=str, default=None,
                        help="Dataset root directory (overrides default based on --dataset)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Raw audio directory (overrides default based on --dataset)")
    parser.add_argument("--use_audio", action="store_true",
                        help="Use raw audio files instead of pre-generated spectrograms")
    parser.add_argument("--htsat_waveform_input", action="store_true",
                        help="Force HTSAT to use waveform input (default: spectrogram preprocessing)")
    parser.add_argument("--range", type=int, nargs=2, default=None,
                        help="Process range of images (e.g., --range 0 10)")
    parser.add_argument("--gpu_batch", type=int, default=200,
                        help="Batch size for LIME predictions (default: 200)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of LIME samples (default: 1000)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: saliency/saliency_lime_{model_type}_{dataset})")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to model checkpoint (overrides default from config)")
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset for images (ResNet50) ---------------------------------------------
class CustomImageDataset(torch.utils.data.Dataset):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.files = []
        for e in self.exts:
            self.files.extend(glob.glob(os.path.join(root_dir, '**', e), recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f'No images found in {root_dir}')
        print(f'Found {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0, path


# ---------------------------------------------------------------------------
# Normalization helpers -----------------------------------------------------
def denormalize_resnet50(t: torch.Tensor) -> np.ndarray:
    """Denormalize ResNet50 tensor to uint8 image (H, W, C).
    Handles both grayscale (1 channel) and RGB (3 channels) automatically.
    """
    # Determine number of channels from tensor shape
    num_channels = t.shape[0]  # (C, H, W)
    
    if num_channels == 1:
        # Grayscale
        mean = [0.5]
        std = [0.5]
        np_img = t.cpu().numpy().transpose(1, 2, 0)  # (1, H, W) -> (H, W, 1)
        np_img = np.array(std) * np_img + np.array(mean)
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
        # Convert to RGB for LIME compatibility (H, W, 3)
        np_img = np.repeat(np_img, 3, axis=2)
    else:
        # RGB
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        np_img = t.cpu().numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        np_img = np.array(std) * np_img + np.array(mean)
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
    
    return np_img


def denormalize_htsat_spectrogram(t: torch.Tensor) -> np.ndarray:
    """Convert HTSAT spectrogram tensor to uint8 image for LIME.
    
    HTSAT spectrograms are (1, mel_bins, time_frames). We convert to
    (mel_bins, time_frames, 3) RGB by repeating the channel.
    """
    # Remove batch dimension if present
    if t.dim() == 4:
        t = t[0]  # (1, 1, H, W) -> (1, H, W)
    if t.dim() == 3:
        t = t[0]  # (1, H, W) -> (H, W)
    
    # Convert to numpy and normalize to [0, 1]
    np_spec = t.cpu().numpy()
    np_spec = (np_spec - np_spec.min()) / (np_spec.max() - np_spec.min() + 1e-6)
    
    # Convert to uint8
    np_spec = (np_spec * 255).astype(np.uint8)
    
    # Convert to RGB by repeating channel (H, W) -> (H, W, 3)
    np_img = np.stack([np_spec] * 3, axis=-1)
    
    return np_img


def normalize_htsat_spectrogram(np_img: np.ndarray, original_min: float, original_max: float) -> torch.Tensor:
    """Convert uint8 image back to HTSAT spectrogram tensor.
    
    Args:
        np_img: uint8 image (H, W, 3) from LIME
        original_min: Original min value for denormalization
        original_max: Original max value for denormalization
    
    Returns:
        Tensor (1, 1, H, W) ready for HTSAT
    """
    # Take first channel (all channels are the same)
    np_spec = np_img[:, :, 0].astype(np.float32) / 255.0
    
    # Denormalize to original range
    np_spec = np_spec * (original_max - original_min) + original_min
    
    # Convert to tensor and add batch/channel dimensions
    t = torch.from_numpy(np_spec).float()
    t = t.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    
    return t


# ---------------------------------------------------------------------------
# Helper: Create preprocessing transform from config -----------------------------
def create_preprocess_transform(config):
    """
    Create preprocessing transform matching the model config.
    Handles both grayscale (1 channel) and RGB (3 channels) based on normalize_mean length.
    
    Args:
        config: Model configuration dict with keys: input_size, normalize_mean, normalize_std
        
    Returns:
        torchvision.transforms.Compose transform
    """
    input_size = config.get('input_size', (224, 224))
    normalize_mean = config.get('normalize_mean', [0.5, 0.5, 0.5])
    normalize_std = config.get('normalize_std', [0.5, 0.5, 0.5])
    
    if len(normalize_mean) == 1:
        # Grayscale (1 channel)
        preprocess_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    else:
        # RGB (3 channels) 
        preprocess_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    
    return preprocess_transform

# ---------------------------------------------------------------------------
# LIME prediction functions -------------------------------------------------
def create_batch_predict_resnet50(model, preprocess_fn):
    """Create batch_predict function for ResNet50."""
    def batch_predict(images: np.ndarray) -> np.ndarray:
        """LIME callback: predict probabilities for a list/array of images.
        
        Parameters
        ----------
        images : np.ndarray
            Shape (N, H, W, 3), uint8 0-255.
        Returns
        -------
        np.ndarray
            Predicted probabilities (N, num_classes).
        """
        model.eval()
        batch = []
        for img in images:
            img = Image.fromarray(img)
            batch.append(preprocess_fn(img))
        batch = torch.stack(batch, dim=0).cuda(non_blocking=True)
        with torch.no_grad():
            preds = model(batch)
        return preds.cpu().numpy()
    
    return batch_predict


def create_batch_predict_htsat(model, original_min, original_max, batch_size: int = 200):
    """Create batch_predict function for HTSAT with internal batching.
    
    This avoids sending all LIME perturbations through the model in one
    giant CUDA batch, which can cause OOM / illegal memory access for
    heavier models like HTSAT on UrbanSound8K.
    """
    def batch_predict(images: np.ndarray) -> np.ndarray:
        """LIME callback: predict probabilities for a list/array of images.
        
        Parameters
        ----------
        images : np.ndarray
            Shape (N, H, W, 3), uint8 0-255 (converted from spectrograms).
        Returns
        -------
        np.ndarray
            Predicted probabilities (N, num_classes).
        """
        model.eval()
        all_preds = []
        n_images = len(images)
        
        for start in range(0, n_images, batch_size):
            batch_images = images[start:start + batch_size]
            batch_tensors = []
            
            for img in batch_images:
                # Convert uint8 image back to spectrogram tensor
                spec_tensor = normalize_htsat_spectrogram(img, original_min, original_max)
                batch_tensors.append(spec_tensor)
            
            batch = torch.cat(batch_tensors, dim=0).cuda(non_blocking=True)
            
            with torch.no_grad():
                preds = model(batch).cpu().numpy()
            all_preds.append(preds)
            
            # Help CUDA recover between large batches
            torch.cuda.empty_cache()
        
        return np.concatenate(all_preds, axis=0)
    
    return batch_predict


# ---------------------------------------------------------------------------
# Main explanation function -------------------------------------------------
def explain_image(model, model_type, img_tensor, preprocessor=None, 
                  num_samples=1000, gpu_batch=200):
    """Run LIME explanation on a single image/spectrogram.
    
    Args:
        model: PyTorch model
        model_type: 'resnet50' or 'htsat'
        img_tensor: Input tensor (already preprocessed and on GPU)
        preprocessor: Preprocessor function for ResNet50 (optional)
        num_samples: Number of LIME samples
        gpu_batch: Batch size for predictions
    
    Returns:
        saliency: Saliency map as numpy array
        predicted_class: Predicted class index
    """
    torch.cuda.empty_cache()
    # Get predicted class (img_tensor is already on GPU)
    with torch.no_grad():
        preds = model(img_tensor)
        predicted_class = preds.argmax(dim=1).item()
    
    # Convert to uint8 numpy for LIME (move to CPU for processing)
    img_tensor_cpu = img_tensor[0].cpu()
    if model_type == 'resnet50':
        orig_img_uint8 = denormalize_resnet50(img_tensor_cpu)
        # Use the passed preprocessor (config-aware) instead of global preprocess
        preprocess_fn = preprocessor if preprocessor is not None else preprocess
        batch_predict = create_batch_predict_resnet50(model, preprocess_fn)
    elif model_type == 'htsat':
        # Store original min/max for denormalization
        if img_tensor_cpu.dim() == 4:
            spec_np = img_tensor_cpu[0].numpy()  # Remove channel dim
        elif img_tensor_cpu.dim() == 3:
            spec_np = img_tensor_cpu.numpy()
        else:
            spec_np = img_tensor_cpu.numpy()
        original_min = spec_np.min()
        original_max = spec_np.max()
        orig_img_uint8 = denormalize_htsat_spectrogram(img_tensor_cpu)
        batch_predict = create_batch_predict_htsat(
            model,
            original_min,
            original_max,
            batch_size=gpu_batch,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmentation_fn = lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
    
    # Run LIME
    explanation = explainer.explain_instance(
        orig_img_uint8,
        batch_predict,
        labels=(predicted_class,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )
    
    # Extract saliency map
    superpixel_weights = dict(explanation.local_exp[predicted_class])  # {seg_id: weight}
    mask = explanation.segments  # (H, W) segment id per pixel
    
    saliency = np.zeros(mask.shape, dtype=np.float32)
    for seg_id, weight in superpixel_weights.items():
        saliency[mask == seg_id] = weight
    
    # Normalize to [0, 1]
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    
    return saliency, predicted_class


# ---------------------------------------------------------------------------
# Main execution -------------------------------------------------------------
def main():
    args = parse_args()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    num_classes = dataset_config['num_classes']
    
    # Set data directories (use user-provided if available, otherwise use defaults)
    datadir = args.datadir if args.datadir is not None else dataset_config['default_spectrogram_dir']
    audio_dir = args.audio_dir if args.audio_dir is not None else dataset_config['default_audio_dir']
    
    # Setup output directory
    if args.output_dir is None:
        out_dir = f'results/{args.model_type}/saliency/saliency_lime_{args.dataset}'
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Experiment Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Dataset: {args.dataset} ({num_classes} classes)")
    print(f"  Spectrogram dir: {datadir}")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Output dir: {out_dir}")
    print()
    
    # Load model configuration (dataset-aware: picks correct weights_path and num_classes)
    config = get_model_config_for_dataset(args.model_type, args.dataset)

    # Create appropriate preprocessing transform based on model config
    # This handles grayscale (1 channel) vs RGB (3 channels) automatically
    if args.model_type == 'resnet50':
        preprocess_transform = create_preprocess_transform(config)
    else:
        preprocess_transform = preprocess  # Use default for HTSAT

    # Determine if HTSAT should use spectrogram mode
    htsat_spectrogram_mode = (args.model_type == 'htsat') and not args.htsat_waveform_input

    # Determine checkpoint path
    weights_path = args.weights_path or config.get('weights_path')
    print(f"Using checkpoint: {weights_path}")
    
    # Load model
    print(f"Loading {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        weights_path=weights_path,
        num_classes=num_classes,
        htsat_spectrogram_mode=htsat_spectrogram_mode
    )
    
    # Determine whether to use audio files (align with RISE framework)
    USE_AUDIO = args.use_audio or args.model_type != 'resnet50'
    
    # Dataset setup
    if USE_AUDIO:
        dataset = AudioDataset(root_dir=audio_dir)
        data_loader = create_audio_data_loader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
        preprocessor = get_preprocessor(
            args.model_type,
            config,
            htsat_spectrogram_mode=htsat_spectrogram_mode
        )
        if htsat_spectrogram_mode:
            print("Using HTSAT spectrogram preprocessing for LIME.")
        else:
            print(f"Using raw audio with {args.model_type} waveform preprocessing")
    else:
        # Use image dataset for ResNet50
        dataset = CustomImageDataset(datadir, transform=preprocess_transform)
        
        # Apply range if specified
        if args.range is not None:
            indices = list(range(args.range[0], min(args.range[1], len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        preprocessor = None
        print(f"Using pre-generated spectrograms with {args.model_type}")
    
    # Process images
    explanations = []
    
    for i, batch in enumerate(tqdm(data_loader, desc='Explaining images')):
        # Extract path and prepare input tensor (align with RISE framework)
        if USE_AUDIO and preprocessor:
            # Audio dataset returns (filepath, label, filepath)
            input_data, _, path = batch
            filepath = input_data[0] if isinstance(input_data, (list, tuple)) else input_data
            path_str = path[0] if isinstance(path, (list, tuple)) else path
            
            # Preprocess audio file to spectrogram (align with RISE: direct .cuda() call)
            img_tensor = preprocessor(filepath).cuda()
        else:
            # Image dataset returns (img, label, path)
            img_tensor, _, path_tensor = batch
            path_str = path_tensor[0]
            img_tensor = img_tensor.cuda()
        
        # Run explanation
        saliency, predicted_class = explain_image(
            model,
            args.model_type,
            img_tensor,
            preprocessor=preprocess_transform if args.model_type == 'resnet50' else None,
            num_samples=args.num_samples,
            gpu_batch=args.gpu_batch
        )
        
        # Save results
        name = os.path.splitext(os.path.basename(path_str))[0]
        
        # Save raw saliency map
        np.save(os.path.join(out_dir, f'{name}_lime.npy'), saliency)
        
        # Save visualization
        # plt.figure(figsize=(10, 5))
        #
        # # Move tensor to CPU for visualization
        # img_tensor_cpu = img_tensor[0].detach().cpu() if img_tensor.is_cuda else img_tensor[0]
        #
        # # Helper function to get 2D spectrogram for display
        # def get_spec_for_display(tensor):
        #     """Convert tensor to 2D numpy array for matplotlib display."""
        #     if tensor.dim() == 4:
        #         # (1, 1, H, W) -> (H, W)
        #         return tensor[0, 0].numpy()
        #     elif tensor.dim() == 3:
        #         # (1, H, W) -> (H, W)
        #         return tensor.squeeze(0).numpy()
        #     else:
        #         return tensor.numpy()
        #
        # # Left: original image
        # plt.subplot(121)
        # plt.axis('off')
        # if args.model_type == 'resnet50':
        #     tensor_imshow(img_tensor_cpu)
        # else:
        #     # For HTSAT, show spectrogram
        #     spec_np = get_spec_for_display(img_tensor_cpu)
        #     plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        # plt.title(get_class_name(predicted_class))
        #
        # # Right: overlay
        # plt.subplot(122)
        # plt.axis('off')
        # if args.model_type == 'resnet50':
        #     tensor_imshow(img_tensor_cpu)
        # else:
        #     spec_np = get_spec_for_display(img_tensor_cpu)
        #     plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        # plt.imshow(saliency, cmap='jet', alpha=0.5)
        #
        # plt.savefig(os.path.join(out_dir, f'{name}_lime.png'),
        #             dpi=150, bbox_inches='tight')
        # plt.close()
        
        explanations.append(saliency)
    
    # Save all explanations
    if explanations:
        explanations_array = np.array(explanations)
        np.save(os.path.join(out_dir, 'all_lime_maps.npy'), explanations_array)
    
    print(f'Done! Saved LIME maps to {out_dir}/')


if __name__ == '__main__':
    main()

