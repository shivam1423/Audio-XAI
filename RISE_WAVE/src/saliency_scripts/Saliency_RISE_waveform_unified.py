#!/usr/bin/env python
# coding: utf-8
"""
RISE Waveform Implementation - UNIFIED for Multiple Models
----------------------------------------------------------
Supports: Wav2Vec2, ACDNet
Original RISE methodology (Petsiuk et al., 2018) adapted for raw audio waveforms.
Random masks with smooth upsampling and multiplicative blending.
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


# ========== MODEL LOADING ==========
def load_wav2vec2_model(checkpoint_path, num_classes):
    """Load Wav2Vec2 model."""
    # Get absolute path to Wav2Vec2 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav2vec2_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Wav2Vec2'))
    
    if wav2vec2_path not in sys.path:
        sys.path.insert(0, wav2vec2_path)
    
    from model.wav2vec2_classifier import Wav2Vec2Classifier
    
    print("Loading Wav2Vec2 model...")
    model = Wav2Vec2Classifier(
        model_name="facebook/wav2vec2-base",
        num_classes=num_classes,
        dropout_rate=0.1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
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


def load_model(model_type, dataset):
    """Load model based on type and dataset."""
    config = get_model_config_for_dataset(model_type, dataset)

    if model_type == 'wav2vec2':
        return load_wav2vec2_model(config['weights_path'], config['num_classes'])
    elif model_type == 'acdnet':
        return load_acdnet_model(
            config['weights_path'],
            config['num_classes'],
            config['sample_rate'],
            config['target_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ========== UNIFIED RISE CLASS ==========
class WaveformRISE:
    """
    RISE explainer for raw audio waveforms - Unified for multiple models.
    
    Generates random binary masks on coarse temporal grid, upsamples to full
    waveform resolution with smoothing, and computes saliency via weighted sum.
    """
    
    def __init__(self, model, model_type, target_length, gpu_batch=50, occlusion='zeros'):
        """
        Initialize WaveformRISE explainer.
        
        Parameters
        ----------
        model : nn.Module
            Model to explain
        model_type : str
            'wav2vec2' or 'acdnet'
        target_length : int
            Audio length in samples
        gpu_batch : int
            Batch size for processing masks
        occlusion : str
            Occlusion method: 'zeros' (silence) or 'gaussian' (noise)
        """
        self.model = model
        self.model_type = model_type
        self.target_length = target_length
        self.gpu_batch = gpu_batch
        self.occlusion = occlusion
        self.masks = None
        self.N = 0
        self.p1 = 0.5
        
    def generate_masks(self, N, n_segments, p1=0.1, soft_masking='linear',
                      edge_sigma=2.0, savepath='masks_waveform.npy'):
        """
        Generate random binary masks (Original RISE methodology).
        
        Parameters
        ----------
        N : int
            Number of masks to generate
        n_segments : int
            Coarse grid resolution (e.g., 100 segments)
        p1 : float
            Probability of keeping each segment (default: 0.1)
        soft_masking : str
            Upsampling method: 'linear', 'step', or 'gaussian'
        edge_sigma : float
            Sigma for gaussian smoothing (in segments)
        savepath : str
            Path to save generated masks
        """
        
        print(f"\n{'='*70}")
        print(f"Generating Waveform RISE Masks - {self.model_type.upper()}")
        print(f"{'='*70}")
        print(f"Total masks: {N}")
        print(f"Coarse grid: {n_segments} segments")
        print(f"Target length: {self.target_length} samples")
        print(f"Segment size: {self.target_length // n_segments} samples/segment")
        print(f"Probability p1: {p1}")
        print(f"Soft masking: {soft_masking}")
        print(f"{'='*70}\n")
        
        cell_size = int(np.ceil(self.target_length / n_segments))
        up_length = (n_segments + 1) * cell_size  # Oversample
        
        masks = []
        
        for i in tqdm(range(N), desc='Generating random binary masks'):
            # Step 1: Generate random binary mask at coarse resolution
            coarse_mask = (np.random.rand(n_segments) < p1).astype(np.float32)
            
            # Step 2: Upsample to oversized length
            mask_upsampled = self._upsample_mask(
                coarse_mask, 
                target_length=up_length,
                method=soft_masking,
                edge_sigma=edge_sigma,
                n_segments=n_segments
            )
            
            # Step 3: Random shift and crop (CRITICAL for RISE)
            shift = np.random.randint(0, cell_size)
            mask_full = mask_upsampled[shift:shift + self.target_length]
            
            masks.append(mask_full)
        
        # Stack into array [N, target_length]
        masks = np.stack(masks, axis=0)
        
        # Reshape to [N, 1, 1, target_length] for consistency
        masks = masks.reshape(N, 1, 1, self.target_length)
        
        # Save to disk
        np.save(savepath, masks)
        print(f"\nSaved {N} masks to {savepath}")
        
        # Load into GPU
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = N
        self.p1 = float(masks.mean())  # Actual mean of generated masks
        print(f"Actual p1 (mean mask value): {self.p1:.4f}\n")
    
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
    
    def _upsample_mask(self, coarse_mask, target_length, method='linear', 
                      edge_sigma=2.0, n_segments=250):
        """
        Upsample coarse mask to full waveform resolution.
        
        Parameters
        ----------
        coarse_mask : np.ndarray
            Shape (n_segments,) - binary mask at coarse resolution
        target_length : int
            Target length in samples
        method : str
            'linear' - linear interpolation (smooth transitions)
            'step' - step function (hard edges)
            'gaussian' - Gaussian smoothing (very smooth)
        edge_sigma : float
            Sigma for gaussian smoothing (in segments)
        n_segments : int
            Number of coarse segments
            
        Returns
        -------
        np.ndarray
            Shape (target_length,) - upsampled mask
        """
        
        if method == 'step':
            # Simple repetition (hard edges)
            segment_size = target_length // n_segments
            mask_full = np.repeat(coarse_mask, segment_size)
            
            # Handle rounding issues
            if len(mask_full) < target_length:
                mask_full = np.pad(mask_full, (0, target_length - len(mask_full)), 
                                 mode='edge')
            elif len(mask_full) > target_length:
                mask_full = mask_full[:target_length]
                
        elif method == 'linear':
            # Linear interpolation (smooth transitions) - Original RISE approach
            x_coarse = np.linspace(0, target_length - 1, len(coarse_mask))
            x_fine = np.arange(target_length)
            mask_full = np.interp(x_fine, x_coarse, coarse_mask)
            
        elif method == 'gaussian':
            # Step + Gaussian blur
            try:
                from scipy.ndimage import gaussian_filter1d
                
                segment_size = target_length // n_segments
                mask_full = np.repeat(coarse_mask, segment_size)
                
                # Handle length mismatch
                if len(mask_full) < target_length:
                    mask_full = np.pad(mask_full, (0, target_length - len(mask_full)), 
                                     mode='edge')
                elif len(mask_full) > target_length:
                    mask_full = mask_full[:target_length]
                
                # Apply Gaussian smoothing
                sigma = edge_sigma * segment_size
                mask_full = gaussian_filter1d(mask_full, sigma=sigma)
            except ImportError:
                print("Warning: scipy not available, falling back to linear interpolation")
                return self._upsample_mask(coarse_mask, target_length, method='linear', 
                                         n_segments=n_segments)
        else:
            raise ValueError(f"Unknown upsampling method: {method}")
        
        return mask_full.astype(np.float32)
    
    def __call__(self, x):
        """
        Apply masks and compute saliency for waveform input.
        
        Parameters
        ----------
        x : torch.Tensor
            Single audio waveform (target_length,) or (1, target_length)
            
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
        
        # Prepare occlusion baseline
        if self.occlusion == 'zeros':
            baseline = torch.zeros_like(x.data)
        elif self.occlusion == 'gaussian':
            # Gaussian noise with same std as input
            std = x.data.std().item()
            baseline = torch.randn_like(x.data) * std
        else:
            baseline = torch.zeros_like(x.data)
        
        # Blend input with baseline using masks (MULTIPLICATIVE - KEY FOR CNNs!)
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
            # Create batch
            if self.model_type == 'wav2vec2':
                batch = stack[i:min(i + self.gpu_batch, N)]
            elif self.model_type == 'acdnet':
                batch = stack[i:min(i + self.gpu_batch, N)]
            
            with torch.no_grad():
                with autocast():
                    if self.model_type == 'wav2vec2':
                        logits = self.model(batch)
                        probs = torch.softmax(logits.float(), dim=1)
                    elif self.model_type == 'acdnet':
                        probs = self.model(batch)  # ACDNet has softmax built-in
            p.append(probs)
        
        p = torch.cat(p)  # [N, num_classes]
        num_classes = p.size(1)
        
        # Compute saliency: weighted sum of masks by predictions
        # p: [N, num_classes]
        # masks: [N, 1, 1, L] -> [N, L]
        masks_flat = self.masks.view(N, L)  # [N, L]
        sal = torch.matmul(p.transpose(0, 1), masks_flat)  # [num_classes, L]
        
        # Normalize by N and p1 (RISE formula)
        sal = sal / (N * self.p1)
        
        return sal


# ========== AUDIO PREPROCESSING ==========
def preprocess_audio(waveform, sample_rate, model_type, config):
    """
    Preprocess audio for specific model.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform
    sample_rate : int
        Current sample rate
    model_type : str
        Model type ('wav2vec2', 'acdnet')
    config : dict
        Model configuration dict from get_model_config_for_dataset()
    
    Returns
    -------
    torch.Tensor
        Preprocessed waveform at target length and sample rate
    """
    target_sr = config['sample_rate']
    target_length = config['target_length']
    
    # Ensure 1D waveform
    if waveform.dim() > 1:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  # stereo -> mono
        else:
            waveform = waveform.squeeze(0)
    waveform = waveform.squeeze()
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Pad or truncate to target length
    current_length = waveform.shape[0]
    
    if current_length < target_length:
        # Pad with zeros
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_length > target_length:
        # Center crop for ACDNet (better for fixed-length models)
        if model_type == 'acdnet':
            start = (current_length - target_length) // 2
            waveform = waveform[start:start + target_length]
        else:
            waveform = waveform[:target_length]
    
    return waveform


def predict_model(model, waveform, model_type):
    """
    Get prediction from model.
    
    Parameters
    ----------
    model : nn.Module
        Model instance
    waveform : torch.Tensor
        Preprocessed waveform (target_length,)
    model_type : str
        Model type
    
    Returns
    -------
    logits : torch.Tensor
        Model logits (1, num_classes)
    probs : torch.Tensor
        Model probabilities (1, num_classes)
    pred_class : int
        Predicted class
    pred_prob : float
        Prediction probability
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        if model_type == 'wav2vec2':
            # Wav2Vec2: expects (batch, samples)
            input_tensor = waveform.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        
        elif model_type == 'acdnet':
            # ACDNet: expects (batch, 1, 1, samples) for Conv2D
            input_tensor = waveform.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            probs = model(input_tensor)  # ACDNet has softmax already
            logits = torch.log(probs + 1e-10)
        
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    return logits, probs, pred_class, pred_prob


# ========== VISUALIZATION ==========
def visualize_results(waveform, saliency_map_norm, filename, output_dir, 
                     pred_class, pred_prob, model_type, config):
    """Generate and save visualizations."""
    target_length = config['target_length']
    sample_rate = config['sample_rate']
    duration = target_length / sample_rate
    
    print("  Generating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original audio waveform
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
    axes[1].plot(time_axis, audio_normalized, color='black', alpha=0.8, linewidth=0.8)
    axes[1].set_title(f'Audio with Saliency Heatmap (RISE {model_type.upper()}) - '
                     f'Class {pred_class} ({pred_prob:.2%})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_rise_{model_type}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved visualization: {filename}_rise_{model_type}.png")


# ========== MAIN SCRIPT ==========
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RISE saliency maps for waveform input - Unified for Multiple Models"
    )
    
    # Dataset and model selection
    parser.add_argument("--dataset", type=str, choices=['esc50', 'urbansound8k'], default='urbansound8k',
                       help="Dataset to use (default: urbansound8k)")
    parser.add_argument("--model", type=str, choices=['wav2vec2', 'acdnet'], default='wav2vec2',
                       help="Model type to use")
    
    # Mask generation parameters
    parser.add_argument("--N", type=int, default=6000,
                       help="Number of masks to generate (default: 6000)")
    parser.add_argument("--n_segments", type=int, default=100,
                       help="Coarse grid resolution in segments (default: 100)")
    parser.add_argument("--p1", type=float, default=0.1,
                       help="Probability for random binary masks (default: 0.1)")
    
    # Soft masking
    parser.add_argument("--soft_masking", type=str, default="linear",
                       choices=["linear", "step", "gaussian"],
                       help="Upsampling method (default: linear)")
    parser.add_argument("--edge_sigma", type=float, default=2.0,
                       help="Sigma for gaussian smoothing in segments (default: 2.0)")
    
    # Occlusion
    parser.add_argument("--occlusion", type=str, default="zeros",
                       choices=["zeros", "gaussian"],
                       help="Occlusion method (default: zeros)")
    
    # Data
    parser.add_argument("--audio_dir", type=str, default=None,
                       help="Directory containing .wav files (default: auto-set from --dataset)")
    
    # Processing
    parser.add_argument("--gpu_batch", type=int, default=50,
                       help="Batch size for processing (default: 50)")
    parser.add_argument("--generate_new", action="store_true",
                       help="Force generation of new masks")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (default: auto-generated)")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Get configuration from central config
    model_type = args.model
    dataset = args.dataset
    config = get_model_config_for_dataset(model_type, dataset)
    target_length = config['target_length']
    sample_rate = config['sample_rate']

    # Set audio directory (auto-determine from dataset if not specified)
    if args.audio_dir is None:
        dataset_config = get_dataset_config(dataset)
        audio_dir = dataset_config['default_audio_dir']
    else:
        audio_dir = args.audio_dir

    # Set output directory
    if args.output_dir is None:
        output_dir = f"results/saliency/saliency_RISE_{model_type}_{dataset}_{args.n_segments}_segments"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/masks", exist_ok=True)

    # Print configuration
    print("=" * 80)
    print("RISE WAVEFORM SALIENCY GENERATION - UNIFIED")
    print("=" * 80)
    print(f"Dataset: {dataset.upper()}")
    print(f"Model: {model_type.upper()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Target length: {target_length} samples")
    print(f"Duration: {target_length / sample_rate:.1f}s")
    print(f"Masks: {args.N}")
    print(f"Segments: {args.n_segments}")
    print(f"Soft masking: {args.soft_masking}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load model
    model = load_model(model_type, dataset)
    
    # Initialize RISE explainer
    explainer = WaveformRISE(
        model=model,
        model_type=model_type,
        target_length=target_length,
        gpu_batch=args.gpu_batch,
        occlusion=args.occlusion
    )
    
    # Generate or load masks
    mask_filename = f"masks_rise_{model_type}_{args.n_segments}seg_{args.soft_masking}.npy"
    maskspath = os.path.join("results/masks", mask_filename)
    
    if args.generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(
            N=args.N,
            n_segments=args.n_segments,
            p1=args.p1,
            soft_masking=args.soft_masking,
            edge_sigma=args.edge_sigma,
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

    for idx, audio_file in enumerate(audio_files, 1):
        start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
        print(f"{'='*70}")
        
        # Load audio
        waveform, orig_sr = torchaudio.load(audio_file)
        
        # Preprocess
        waveform = preprocess_audio(waveform, orig_sr, model_type, config)
        print(f"  Waveform shape: {waveform.shape}")
        
        # Get prediction
        logits, probs, pred_class, pred_prob = predict_model(model, waveform, model_type)
        print(f"  Predicted class: {pred_class}, Probability: {pred_prob:.4f}")
        
        # Generate RISE explanation
        print("\n  === Generating RISE Saliency Map ===")
        saliency_maps = explainer(waveform.cuda())  # [num_classes, target_length]
        saliency_map = saliency_maps[pred_class].cpu().numpy()  # [target_length]
        
        # Normalize to [0, 1]
        saliency_map_norm = saliency_map - saliency_map.min()
        if saliency_map_norm.max() > 0:
            saliency_map_norm /= saliency_map_norm.max()
        
        # Save results
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        np.save(os.path.join(output_dir, f'{filename}_rise_{model_type}.npy'), 
               saliency_map_norm)
        print(f"  ✓ Saved saliency map: {filename}_rise_{model_type}.npy")
        
        # Visualization
        visualize_results(waveform, saliency_map_norm, filename, output_dir,
                         pred_class, pred_prob, model_type, config)
        
        print(f"  RISE computation took {time.time() - start_time:.2f} seconds")
    
    # Print summary
    print("\n" + "="*70)
    print(f"=== RISE Waveform Processing Complete ({model_type.upper()}) ===")
    print(f"Processed {len(audio_files)} files")
    print(f"Results saved to {output_dir}/")
    print(f"Masks saved to {maskspath}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()



