#!/usr/bin/env python
# coding: utf-8
"""
RISE Waveform Mask & Occlusion Framework for Audio Saliency - Unified
---------------------------------------------------------------------
Enhanced RISE implementation with structured temporal masks and multiple occlusion strategies.
Supports: Wav2Vec2, ACDNet

Mask Types:
  - Contiguous segments: Long temporal windows (10-500ms)
  - Scattered patches: Multiple short temporal bursts (5-50ms)

Soft Masking:
  - Discrete (hard edges)
  - Gaussian smoothing
  - Bilinear interpolation

Occlusion Strategies:
  - Zeros (silence)
  - Gaussian noise
"""

import torch
import torch.nn as nn
import torchaudio
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
import glob
import argparse
import time
# Import base RISE class
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.explanations import RISE

# ========== MODEL CONFIGURATIONS (inline — avoids caching RISE_WAVE config in sys.modules) ==========
def get_model_config_for_dataset(model_type, dataset='esc50'):
    """Get model configuration for specific dataset."""
    base_configs = {
        'wav2vec2': {
            'sample_rate': 16000,
            'target_length': 80000 if dataset == 'esc50' else 64000,
            'model_type': 'wav2vec2'
        },
        'acdnet': {
            'sample_rate': 20000,
            'target_length': 30225,
            'model_type': 'acdnet'
        }
    }
    config = base_configs[model_type].copy()
    config['num_classes'] = 50 if dataset == 'esc50' else 10
    if model_type == 'wav2vec2':
        config['weights_path'] = 'checkpoints/best_model_wav2vec2_esc50.pt' if dataset == 'esc50' else 'checkpoints/best_model_wav2vec2_us8k.pt'
    elif model_type == 'acdnet':
        config['weights_path'] = 'checkpoints/acdnet_esc50.pt' if dataset == 'esc50' else 'checkpoints/acdnet_us8k_best.pt'
    return config


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    if dataset_name == 'esc50':
        return {'num_classes': 50, 'default_audio_dir': '../ESC50/audio'}
    elif dataset_name == 'urbansound8k':
        return {'num_classes': 10, 'default_audio_dir': '../UrbanSound8K/audio/fold10'}
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")


def load_acdnet_model(checkpoint_path, num_classes, sample_rate, target_length):
    """Load ACDNet model."""
    # Get absolute path to ACDNet directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ACDNet_UrbanSound8K'))
    from models.acdnet import ACDNetV2
    
    print("Loading ACDNet model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    
    # Create model using ACDNetV2
    model = ACDNetV2(target_length, num_classes, sample_rate).to(device)
    
    # Load weights (handle different checkpoint formats)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'weight' in state:
        model.load_state_dict(state['weight'])
    else:
        model.load_state_dict(state)
    
    model.eval()
    
    # Wrap with DataParallel if CUDA available
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    
    for p in model.parameters():
        p.requires_grad = False
    
    print("✓ ACDNet model loaded")
    return model


def load_model(model_type, dataset, checkpoint_path=None):
    """Load model based on type and dataset.

    Args:
        model_type: 'wav2vec2' or 'acdnet'
        dataset: 'esc50' or 'urbansound8k'
        checkpoint_path: Optional override for the checkpoint path from config.
    """
    config = get_model_config_for_dataset(model_type, dataset)
    if checkpoint_path is not None:
        config['weights_path'] = checkpoint_path
    checkpoint_path = config['weights_path']
    
    if model_type == 'wav2vec2':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wav2vec2_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Wav2Vec2'))

        if wav2vec2_path not in sys.path:
            sys.path.insert(0, wav2vec2_path)

        from model.wav2vec2_classifier import Wav2Vec2Classifier

        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2Classifier(
            model_name="facebook/wav2vec2-base",
            num_classes=config['num_classes'],
            dropout_rate=0.1
        )
        
        checkpoint = torch.load(checkpoint_path, 
                              map_location="cuda" if torch.cuda.is_available() else "cpu")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
        print("✓ Wav2Vec2 model loaded")
        return model
        
    elif model_type == 'acdnet':
        print("loading acdnet model")
        return load_acdnet_model(
            checkpoint_path,
            config['num_classes'],
            config['sample_rate'],
            config['target_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class WaveformRISE_Mask_Occlusion_Framework(RISE):
    """
    Enhanced RISE explainer for raw audio waveforms with structured temporal masks.
    
    Extends base RISE class to handle 1D audio with:
    - Structured mask types (contiguous segments, scattered patches)
    - Multiple soft masking strategies (discrete, gaussian, bilinear)
    - Enhanced occlusion baselines (zeros, gaussian)
    Supports: Wav2Vec2, ACDNet
    """
    
    def __init__(self, model, model_type, target_length, gpu_batch=50, occlusion='zeros', 
                 sample_rate=16000, soft_masking='discrete'):
        """
        Initialize enhanced WaveformRISE explainer.
        
        Parameters
        ----------
        model : nn.Module
            Model to explain (Wav2Vec2 or ACDNet)
        model_type : str
            'wav2vec2' or 'acdnet'
        target_length : int
            Audio length in samples (e.g., 80000 for 5s @ 16kHz, 30225 for 1.5s @ 20kHz)
        gpu_batch : int
            Batch size for processing masks
        occlusion : str
            Occlusion method: 'zeros' or 'gaussian'
        sample_rate : int
            Audio sample rate
        soft_masking : str
            Soft masking method: 'discrete', 'gaussian', or 'bilinear'
        """
        # Pass dummy input_size for base class compatibility
        super().__init__(model, input_size=(1, target_length), gpu_batch=gpu_batch)
        self.model_type = model_type
        self.target_length = target_length
        self.occlusion = occlusion
        self.sample_rate = sample_rate
        self.soft_masking = soft_masking
        self.p1 = None
        
    def generate_masks(self, N, contiguous_frac=0.5, scattered_frac=0.5,
                      segment_duration_ms=(10, 250), patch_duration_ms=(5, 50),
                      patch_count_range=(1, 10), soft_masking='discrete',
                      edge_sigma_samples=160, savepath='masks_waveform_temporal.npy',
                      n_segments=None, p1=None, edge_sigma=None):
        """
        Generate structured temporal masks with soft masking options.
        
        Parameters
        ----------
        N : int
            Number of masks to generate
        contiguous_frac : float
            Fraction of contiguous segment masks (default: 0.5)
        scattered_frac : float
            Fraction of scattered patch masks (default: 0.5)
        segment_duration_ms : tuple
            (min, max) duration for contiguous segments in ms
        patch_duration_ms : tuple
            (min, max) duration for scattered patches in ms
        patch_count_range : tuple
            (min, max) number of patches per mask
        soft_masking : str
            'discrete' (hard edges), 'gaussian', or 'bilinear'
        edge_sigma_samples : int
            Sigma for Gaussian smoothing in samples (default: 160 = 10ms @ 16kHz)
        savepath : str
            Path to save generated masks
        n_segments : int (optional)
            For compatibility with original interface
        p1 : float (optional)
            For compatibility with original interface
        edge_sigma : float (optional)
            For compatibility with original interface
        """
        
        # Update soft_masking if provided
        if soft_masking is not None:
            self.soft_masking = soft_masking
        
        # Convert durations to samples
        seg_min_samples = int(segment_duration_ms[0] * self.sample_rate / 1000)
        seg_max_samples = int(segment_duration_ms[1] * self.sample_rate / 1000)
        patch_min_samples = int(patch_duration_ms[0] * self.sample_rate / 1000)
        patch_max_samples = int(patch_duration_ms[1] * self.sample_rate / 1000)
        
        print(f"\n{'='*70}")
        print(f"Generating Temporal Structured RISE Masks - {self.model_type.upper()}")
        print(f"{'='*70}")
        print(f"Total masks: {N}")
        print(f"Target length: {self.target_length} samples ({self.target_length/self.sample_rate:.2f}s)")
        print(f"Mask distribution:")
        print(f"  - Contiguous segments: {int(N * contiguous_frac)} masks "
              f"({segment_duration_ms[0]}-{segment_duration_ms[1]}ms)")
        print(f"  - Scattered patches: {int(N * scattered_frac)} masks "
              f"({patch_duration_ms[0]}-{patch_duration_ms[1]}ms, {patch_count_range[0]}-{patch_count_range[1]} patches)")
        print(f"Soft masking: {self.soft_masking}")
        if self.soft_masking == 'gaussian':
            print(f"  Edge sigma: {edge_sigma_samples} samples (~{1000*edge_sigma_samples/self.sample_rate:.1f}ms)")
        print(f"{'='*70}\n")
        
        # Determine counts for each strategy
        n_contiguous = int(N * contiguous_frac)
        n_scattered = N - n_contiguous
        
        masks = []
        
        # Generate contiguous segment masks
        for _ in tqdm(range(n_contiguous), desc='Generating contiguous segment masks'):
            mask = self._generate_contiguous_mask(
                seg_min_samples, seg_max_samples
            )
            masks.append(mask)
        
        # Generate scattered patch masks
        for _ in tqdm(range(n_scattered), desc='Generating scattered patch masks'):
            mask = self._generate_scattered_mask(
                patch_min_samples, patch_max_samples, patch_count_range
            )
            masks.append(mask)
        
        # Stack and apply soft masking
        print("Applying soft masking...")
        for i in tqdm(range(len(masks)), desc=f'Applying {self.soft_masking} masking'):
            if self.soft_masking == 'gaussian':
                masks[i] = self._apply_gaussian_smoothing(masks[i], edge_sigma_samples)
            elif self.soft_masking == 'bilinear':
                masks[i] = self._apply_bilinear_upsampling(masks[i])
            # For 'discrete', no processing needed
        
        # Stack into array [N, target_length]
        masks = np.stack(masks, axis=0)
        
        # Reshape to [N, 1, 1, target_length] for model input
        masks = masks.reshape(N, 1, 1, self.target_length)
        
        # Save to disk
        np.save(savepath, masks)
        print(f"\nSaved {N} masks to {savepath}")
        
        # Load into GPU
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = N
        self.p1 = float(masks.mean())
        print(f"Actual p1 (mean mask value): {self.p1:.4f}\n")
    
    def _generate_contiguous_mask(self, seg_min_samples, seg_max_samples):
        """
        Generate a mask with one or a few long contiguous segments.
        
        Parameters
        ----------
        seg_min_samples : int
            Minimum segment length in samples
        seg_max_samples : int
            Maximum segment length in samples
            
        Returns
        -------
        np.ndarray
            Shape (target_length,) - binary mask
        """
        mask = np.zeros(self.target_length, dtype=np.float32)
        
        # Randomly choose 1-3 segments
        n_segments = np.random.randint(1, 4)
        
        for _ in range(n_segments):
            seg_length = np.random.randint(seg_min_samples, seg_max_samples + 1)
            seg_length = min(seg_length, self.target_length)
            
            # Random start position
            max_start = max(0, self.target_length - seg_length)
            if max_start > 0:
                start_pos = np.random.randint(0, max_start)
            else:
                start_pos = 0
            
            mask[start_pos:start_pos + seg_length] = 1.0
        
        return mask
    
    def _generate_scattered_mask(self, patch_min_samples, patch_max_samples, 
                                 patch_count_range):
        """
        Generate a mask with multiple short scattered patches.
        
        Parameters
        ----------
        patch_min_samples : int
            Minimum patch length in samples
        patch_max_samples : int
            Maximum patch length in samples
        patch_count_range : tuple
            (min, max) number of patches
            
        Returns
        -------
        np.ndarray
            Shape (target_length,) - binary mask
        """
        mask = np.zeros(self.target_length, dtype=np.float32)
        
        # Random number of patches
        n_patches = np.random.randint(patch_count_range[0], patch_count_range[1] + 1)
        
        for _ in range(n_patches):
            patch_length = np.random.randint(patch_min_samples, patch_max_samples + 1)
            patch_length = min(patch_length, self.target_length)
            
            # Random start position
            max_start = max(0, self.target_length - patch_length)
            if max_start > 0:
                start_pos = np.random.randint(0, max_start)
            else:
                start_pos = 0
            
            mask[start_pos:start_pos + patch_length] = 1.0
        
        return mask
    
    def _apply_gaussian_smoothing(self, mask, sigma_samples):
        """
        Apply Gaussian smoothing to soften mask edges.
        
        Parameters
        ----------
        mask : np.ndarray
            Shape (target_length,) - binary mask
        sigma_samples : int
            Gaussian sigma in samples
            
        Returns
        -------
        np.ndarray
            Smoothed mask
        """
        try:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(mask.astype(np.float32), sigma=sigma_samples)
        except ImportError:
            print("Warning: scipy not available, using discrete masking")
            return mask.astype(np.float32)
    
    def _apply_bilinear_upsampling(self, mask):
        """
        Apply bilinear upsampling (original RISE approach).
        Downsample to coarse grid then upsample with linear interpolation.
        
        Parameters
        ----------
        mask : np.ndarray
            Shape (target_length,) - binary mask
            
        Returns
        -------
        np.ndarray
            Smoothed mask via bilinear upsampling
        """
        # Downsample factor (like original RISE)
        downsample_factor = 8
        coarse_length = max(1, self.target_length // downsample_factor)
        
        # Downsample: average over windows
        coarse_mask = np.zeros(coarse_length, dtype=np.float32)
        window_size = self.target_length / coarse_length
        
        for i in range(coarse_length):
            start_idx = int(i * window_size)
            end_idx = int((i + 1) * window_size)
            coarse_mask[i] = mask[start_idx:end_idx].mean()
        
        # Upsample with linear interpolation
        x_coarse = np.linspace(0, self.target_length - 1, coarse_length)
        x_fine = np.arange(self.target_length)
        smooth_mask = np.interp(x_fine, x_coarse, coarse_mask)
        
        return smooth_mask.astype(np.float32)
    
    def load_masks(self, filepath):
        """Load pre-generated masks from file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Masks file not found: {filepath}")
        
        masks = np.load(filepath)
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = float(self.masks.mean().item())
        print(f"Loaded {self.N} masks from {filepath}")
        print(f"Mean mask value (p1): {self.p1:.4f}")
    
    def forward(self, x):
        """
        Apply masks with enhanced occlusion strategies and compute saliency.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape [1, target_length] or [target_length] - single audio waveform
            
        Returns
        -------
        torch.Tensor
            Shape [num_classes, target_length] - saliency map for each class
        """
        from torch.cuda.amp import autocast
        
        N = self.N
        L = self.target_length
        
        # Ensure input has correct shape [1, 1, 1, L]
        if x.dim() == 2:  # [1, L]
            x = x.unsqueeze(1).unsqueeze(1)  # [1, 1, 1, L]
        elif x.dim() == 1:  # [L]
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, L]
        
        # Prepare occlusion baseline based on strategy
        if self.occlusion == 'zeros':
            baseline = torch.zeros_like(x.data)
            
        elif self.occlusion == 'gaussian':
            # Gaussian noise with same std as input
            std = x.data.std().item()
            baseline = torch.randn_like(x.data) * std
            
        else:
            baseline = torch.zeros_like(x.data)
        
        # Blend input with baseline using masks
        # self.masks: [N, 1, 1, L]
        # x: [1, 1, 1, L]
        # Result: [N, 1, 1, L]
        stack = self.masks * x.data + (1.0 - self.masks) * baseline
        
        # Model-specific shape handling
        if self.model_type == 'wav2vec2':
            # Squeeze to [N, L] for Wav2Vec2 input
            stack = stack.squeeze(1).squeeze(1)  # [N, L]
        elif self.model_type == 'acdnet':
            # Keep [N, 1, 1, L] for ACDNet's Conv2D
            pass  # stack is already correct shape
        
        # Run model in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            # Create stack for this batch only
            batch_masks = self.masks[i:min(i + self.gpu_batch, N)]
            batch_stack = batch_masks * x.data + (1.0 - batch_masks) * baseline
            
            # Model-specific shape handling for batch
            if self.model_type == 'wav2vec2':
                batch_stack = batch_stack.squeeze(1).squeeze(1)  # [batch_size, L]
            elif self.model_type == 'acdnet':
                # Keep [batch_size, 1, 1, L] for ACDNet
                pass
            
            with torch.no_grad():
                with autocast():
                    if self.model_type == 'wav2vec2':
                        logits = self.model(batch_stack)
                        probs = torch.softmax(logits.float(), dim=1)
                    elif self.model_type == 'acdnet':
                        probs = self.model(batch_stack)  # ACDNet has softmax built-in
            p.append(probs)
            
            # Free memory
            del batch_stack, batch_masks
            torch.cuda.empty_cache()
        
        p = torch.cat(p)  # [N, num_classes]
        num_classes = p.size(1)
        
        # Compute saliency
        masks_flat = self.masks.view(N, L)
        sal = torch.matmul(p.transpose(0, 1), masks_flat)
        sal = sal / (N * self.p1)
        
        return sal


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RISE saliency maps for waveform input with structured masks and occlusion strategies - Unified for Wav2Vec2 and ACDNet"
    )
    
    # Dataset and model selection
    parser.add_argument("--dataset", type=str, choices=['esc50', 'urbansound8k'], default='urbansound8k',
                       help="Dataset to use (default: urbansound8k)")
    parser.add_argument("--model", type=str, default="wav2vec2",
                       choices=["wav2vec2", "acdnet"],
                       help="Model type: 'wav2vec2' or 'acdnet' (default: wav2vec2)")
    
    # Mask type selection
    parser.add_argument("--mask_type", type=str, default="all",
                       choices=["all", "contiguous", "scattered"],
                       help="Mask strategy: all (mixed), contiguous, or scattered (default: all)")
    
    # Mask generation parameters
    parser.add_argument("--N", type=int, default=6000,
                       help="Number of masks to generate (default: 6000)")
    parser.add_argument("--contiguous_frac", type=float, default=0.5,
                       help="Fraction of contiguous segment masks (default: 0.5)")
    parser.add_argument("--scattered_frac", type=float, default=0.5,
                       help="Fraction of scattered patch masks (default: 0.5)")
    
    # Temporal parameters
    parser.add_argument("--segment_duration_ms", type=str, default="10,500",
                       help="Min,max duration for contiguous segments in ms (default: 10,500)")
    parser.add_argument("--patch_duration_ms", type=str, default="5,50",
                       help="Min,max duration for scattered patches in ms (default: 5,50)")
    parser.add_argument("--patch_count_range", type=str, default="1,10",
                       help="Min,max number of patches per mask (default: 1,10)")
    
    # Soft masking
    parser.add_argument("--soft_masking", type=str, default="discrete",
                       choices=["discrete", "gaussian", "bilinear"],
                       help="Soft masking method (default: discrete)")
    parser.add_argument("--edge_sigma_ms", type=float, default=10.0,
                       help="Gaussian edge sigma in ms (default: 10.0)")
    
    # Occlusion
    parser.add_argument("--occlusion", type=str, default="zeros",
                       choices=["zeros", "gaussian"],
                       help="Occlusion method (default: zeros)")
    
    # Data & Model
    parser.add_argument("--audio_dir", type=str, default=None,
                       help="Directory containing .wav files (default: auto-set from --dataset)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint (default: auto-determined based on model type)")
    parser.add_argument("--target_length", type=int, default=None,
                       help="Audio length in samples (default: auto-determined based on model)")
    parser.add_argument("--sample_rate", type=int, default=None,
                       help="Audio sample rate (default: auto-determined based on model)")
    
    # Processing
    parser.add_argument("--gpu_batch", type=int, default=50,
                       help="Batch size for processing (default: 50)")
    parser.add_argument("--generate_new", action="store_true",
                       help="Force generation of new masks")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (default: auto-generated)")
    parser.add_argument("--mask_name", type=str, default=None,
                       help="Custom mask filename (default: auto-generated)")
    
    # Legacy compatibility
    parser.add_argument("--n_segments", type=int, default=100,
                       help="(Legacy) Coarse grid resolution - not used in this framework")
    parser.add_argument("--p1", type=float, default=0.5,
                       help="(Legacy) Probability parameter - not used in this framework")
    parser.add_argument("--edge_sigma", type=float, default=2.0,
                       help="(Legacy) Edge sigma - use edge_sigma_ms instead")
    
    return parser.parse_args()



def main():
    """Main execution function."""
    args = parse_args()
    
    # Get model configuration from central config
    model_type = args.model
    dataset = args.dataset
    config = get_model_config_for_dataset(model_type, dataset)

    # Apply CLI overrides on top of config (explicit args take precedence)
    sample_rate = args.sample_rate if args.sample_rate is not None else config['sample_rate']
    target_length = args.target_length if args.target_length is not None else config['target_length']

    # Set audio directory (auto-determine from dataset if not specified)
    if args.audio_dir is None:
        dataset_cfg = get_dataset_config(dataset)
        audio_dir = dataset_cfg['default_audio_dir']
    else:
        audio_dir = args.audio_dir
    
    # Parse string arguments to tuples
    segment_duration_ms = tuple(map(int, args.segment_duration_ms.split(',')))
    patch_duration_ms = tuple(map(int, args.patch_duration_ms.split(',')))
    patch_count_range = tuple(map(int, args.patch_count_range.split(',')))
    
    # Determine mask type fractions based on mask_type
    if args.mask_type == 'contiguous':
        contiguous_frac, scattered_frac = 1.0, 0.0
    elif args.mask_type == 'scattered':
        contiguous_frac, scattered_frac = 0.0, 1.0
    else:  # 'all'
        contiguous_frac = args.contiguous_frac
        scattered_frac = args.scattered_frac
    
    # Normalize fractions
    total_frac = contiguous_frac + scattered_frac
    if total_frac > 0:
        contiguous_frac /= total_frac
        scattered_frac /= total_frac
    
    # Convert edge sigma from ms to samples
    edge_sigma_samples = int(args.edge_sigma_ms * sample_rate / 1000)

    # Set output directory
    mask_suffix = f"_{args.mask_type}" if args.mask_type != "all" else "_combined"
    soft_suffix = f"_{args.soft_masking}" if args.soft_masking != "discrete" else ""
    occlusion_suffix = f"_occlusion_{args.occlusion}"

    if args.output_dir is None:
        output_dir = f"results/saliency/saliency_RISE_waveform_{model_type}_{dataset}{mask_suffix}{soft_suffix}{occlusion_suffix}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/masks", exist_ok=True)

    # Load model
    print("\n" + "="*70)
    print(f"Loading {model_type.upper()} model...")
    print("="*70)
    model = load_model(model_type, dataset, args.checkpoint_path)

    print("Model loaded successfully!\n")
    print(f"Configuration:")
    print(f"  Dataset: {dataset.upper()}")
    print(f"  Model: {model_type.upper()}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Target length: {target_length} samples ({target_length / sample_rate:.2f}s)")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Soft masking: {args.soft_masking}")
    print(f"  Occlusion: {args.occlusion}")
    print(f"  GPU batch size: {args.gpu_batch}\n")

    # Initialize RISE explainer
    explainer = WaveformRISE_Mask_Occlusion_Framework(
        model=model,
        model_type=model_type,
        target_length=target_length,
        gpu_batch=args.gpu_batch,
        occlusion=args.occlusion,
        sample_rate=sample_rate,
        soft_masking=args.soft_masking
    )
    
    # Generate or load masks
    if args.mask_name is None:
        mask_filename = f"masks_waveform_{model_type}{mask_suffix}{soft_suffix}.npy"
    else:
        mask_filename = f"{args.mask_name}.npy"
    
    maskspath = os.path.join("results/masks", mask_filename)
    
    if args.generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(
            N=args.N,
            contiguous_frac=contiguous_frac,
            scattered_frac=scattered_frac,
            segment_duration_ms=segment_duration_ms,
            patch_duration_ms=patch_duration_ms,
            patch_count_range=patch_count_range,
            soft_masking=args.soft_masking,
            edge_sigma_samples=edge_sigma_samples,
            savepath=maskspath
        )
    else:
        print(f"\nLoading existing masks from {maskspath}...")
        explainer.load_masks(maskspath)
        print()
    
    # Get all audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"Found {len(audio_files)} audio files in {audio_dir}\n")

    if len(audio_files) == 0:
        print(f"No .wav files found in {audio_dir}")
        return
    
    # Process each audio file
    print("="*70)
    print("Processing audio files...")
    print("="*70 + "\n")
    
    skipped_count = 0
    processed_count = 0

    for idx, audio_file in enumerate(audio_files, 1):
        start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
        print(f"{'='*70}")
        
        # Check if saliency map already exists
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        saliency_npy_path = os.path.join(output_dir, f'{filename}_rise_{model_type}.npy')
        saliency_png_path = os.path.join(output_dir, f'{filename}_rise_{model_type}.png')
        
        if os.path.exists(saliency_npy_path) and os.path.exists(saliency_png_path):
            print(f"✓ Saliency map already exists, skipping...")
            skipped_count += 1
            continue
        
        # Load audio
        waveform, orig_sr = torchaudio.load(audio_file)
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze()

        # Pad or trim to target length
        if waveform.shape[0] < target_length:
            waveform = torch.nn.functional.pad(waveform,
                                               (0, target_length - waveform.shape[0]))
        elif waveform.shape[0] > target_length:
            # Center crop for ACDNet (better for fixed-length models)
            if model_type == 'acdnet':
                start = (waveform.shape[0] - target_length) // 2
                waveform = waveform[start:start + target_length]
            else:
                waveform = waveform[:target_length]
        
        print(f"Waveform shape: {waveform.shape}")
        
        # Get prediction
        with torch.no_grad():
            if model_type == 'wav2vec2':
                logits = model(waveform.unsqueeze(0).cuda())
                probs = torch.softmax(logits, dim=1)
            elif model_type == 'acdnet':
                probs = model(waveform.unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda())
                logits = torch.log(probs + 1e-10)
            
            pred_class = probs.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        print(f"Predicted class: {pred_class}, Probability: {pred_prob:.4f}")
        
        # Generate RISE explanation
        print("\n=== Generating RISE Saliency Map ===")
        saliency_maps = explainer(waveform.cuda())  # [num_classes, target_length]
        saliency_map = saliency_maps[pred_class].cpu().numpy()  # [target_length]
        
        # Normalize to [0, 1]
        saliency_map_norm = saliency_map - saliency_map.min()
        if saliency_map_norm.max() > 0:
            saliency_map_norm /= saliency_map_norm.max()
        
        # Save results
        np.save(saliency_npy_path, saliency_map_norm)
        
        # Visualization (LIME style - vertical layout)
        print("Generating visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Original audio waveform
        duration = target_length / sample_rate
        time_axis = np.linspace(0, duration, target_length)
        audio_np = waveform.cpu().numpy()

        axes[0].plot(time_axis, audio_np, linewidth=0.5, color='black')
        axes[0].set_title(f'Original Audio Waveform ({model_type.upper()})')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlabel('Time (s)')
        axes[0].grid(True, alpha=0.3)

        # Audio with saliency heatmap overlay
        saliency_2d = np.repeat(saliency_map_norm.reshape(1, -1), 100, axis=0)
        audio_normalized = audio_np / (np.max(np.abs(audio_np)) + 1e-8)

        axes[1].imshow(saliency_2d, aspect='auto', cmap='jet',
                      extent=[0, duration, -1, 1],
                      interpolation='bilinear', alpha=0.6)
        axes[1].plot(time_axis, audio_normalized, color='black', alpha=0.8,
                    linewidth=0.8)
        axes[1].set_title(f'Audio with Saliency Heatmap Overlay (RISE {model_type.upper()}) - '
                         f'Class {pred_class} ({pred_prob:.2%})')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_ylim(-1, 1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(saliency_png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved visualization to {saliency_png_path}")
        print(f"RISE computation took {time.time() - start_time:.2f} seconds")
        processed_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print(f"=== RISE Waveform Mask & Occlusion Framework Complete ({model_type.upper()}) ===")
    print(f"{'='*70}")
    print(f"Total audio files: {len(audio_files)}")
    print(f"  - Processed: {processed_count}")
    print(f"  - Skipped (already exist): {skipped_count}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_type.upper()}")
    print(f"  Mask type: {args.mask_type}")
    print(f"    - Contiguous: {contiguous_frac*100:.1f}%")
    print(f"    - Scattered: {scattered_frac*100:.1f}%")
    print(f"  Soft masking: {args.soft_masking}")
    print(f"  Occlusion: {args.occlusion}")
    print(f"\nOutput:")
    print(f"  Results: {output_dir}/")
    print(f"  Masks: {maskspath}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

