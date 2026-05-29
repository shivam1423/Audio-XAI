#!/usr/bin/env python
# coding: utf-8
"""
LIME Waveform Implementation - UNIFIED for Multiple Models
----------------------------------------------------------
Supports: Wav2Vec2, ACDNet
Direct control over segment perturbations (zero-out)
"""

import torch
import torch.nn as nn
import torchaudio
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
import argparse
from sklearn.linear_model import Ridge
import glob

# ========== MODEL CONFIGURATIONS ==========
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
        if dataset == 'esc50':
            config['weights_path'] = "checkpoints/best_model_wav2vec2_esc50.pt"
        else:
            config['weights_path'] = 'checkpoints/best_model_wav2vec2_us8k.pt'
    elif model_type == 'acdnet':
        if dataset == 'esc50':
            config['weights_path'] = 'checkpoints/acdnet_esc50.pt'
        else:
            config['weights_path'] = 'checkpoints/acdnet_us8k_best.pt'

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
    print("✓ Wav2Vec2 model loaded")
    return model


def load_acdnet_model(checkpoint_path, num_classes, sample_rate, target_length):
    """Load ACDNet model."""
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
        Model configuration
    
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
            # Average channels to convert stereo to mono
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
    
    # Resample if needed
    if sample_rate != target_sr:
        print(f"  Resampling: {sample_rate}Hz → {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Pad or truncate to target length
    current_length = waveform.shape[0]
    
    if current_length < target_length:
        # Pad with zeros
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        print(f"  Padded: {current_length} → {target_length} samples")
    elif current_length > target_length:
        # Center crop for ACDNet (better for fixed-length models)
        # Regular crop for Wav2Vec2
        if model_type == 'acdnet':
            start = (current_length - target_length) // 2
            waveform = waveform[start:start + target_length]
            print(f"  Center cropped: {current_length} → {target_length} samples")
        else:
            waveform = waveform[:target_length]
            print(f"  Cropped: {current_length} → {target_length} samples")
    else:
        print(f"  Length OK: {target_length} samples")
    
    return waveform


# ========== MODEL PREDICTION ==========
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
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'wav2vec2':
        # Wav2Vec2: expects (batch, samples)
        input_tensor = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
    
    elif model_type == 'acdnet':
        # ACDNet: expects (batch, 1, 1, samples) for Conv2D
        input_tensor = waveform.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = model(input_tensor)  # ACDNet has softmax already
            logits = torch.log(probs + 1e-10)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return logits, probs


# ========== LIME EXPLANATION (MODEL-AGNOSTIC) ==========
def lime_audio(waveform, model, pred_class, model_type, n_segments=250, num_samples=1000):
    """
    LIME Waveform implementation for audio segments.
    Works with any model through unified predict_model() interface.
    
    Parameters
    ----------
    waveform : torch.Tensor
        Shape (target_length,) - preprocessed audio waveform
    model : nn.Module
        Model instance
    pred_class : int
        Class to explain
    model_type : str
        Model type ('wav2vec2', 'acdnet')
    n_segments : int
        Number of segments (default: 250)
    num_samples : int
        Number of perturbations (default: 1000)
    
    Returns
    -------
    np.ndarray
        Importance scores for each segment (n_segments,)
    """
    segment_size = len(waveform) // n_segments
    
    # Generate random binary masks (0 = zero out segment, 1 = keep segment)
    print(f"  Generating {num_samples} random perturbations...")
    masks = np.random.randint(0, 2, size=(num_samples, n_segments))
    
    # Get predictions for each perturbation
    print(f"  Evaluating perturbations...")
    predictions = []
    
    # Process in batches for efficiency
    batch_size = 50
    for batch_start in tqdm(range(0, num_samples, batch_size), desc="  Processing batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_masks = masks[batch_start:batch_end]
        
        batch_waveforms = []
        for mask in batch_masks:
            # Create perturbed waveform
            perturbed = waveform.clone()

            for seg_id in range(n_segments):
                if mask[seg_id] == 0:  # Zero out this segment
                    start = seg_id * segment_size
                    end = start + segment_size
                    perturbed[start:end] = 0
            batch_waveforms.append(perturbed)
        
        # Batch prediction using unified interface
        batch_tensor = torch.stack(batch_waveforms)
        
        # Get predictions (model-specific handling inside predict_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'wav2vec2':
            # Wav2Vec2 can handle batch directly
            batch_tensor = batch_tensor.to(device)
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                batch_preds = probs[:, pred_class].cpu().numpy()
        
        elif model_type == 'acdnet':
            # ACDNet needs (batch, 1, 1, samples) shape
            batch_tensor = batch_tensor.unsqueeze(1).unsqueeze(1).to(device)
            with torch.no_grad():
                probs = model(batch_tensor)
                batch_preds = probs[:, pred_class].cpu().numpy()
        
        predictions.extend(batch_preds)
    
    predictions = np.array(predictions)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Compute distances from original (all segments ON = all ones)
    original_mask = np.ones(n_segments)
    distances = np.array([
        np.linalg.norm(mask - original_mask) for mask in masks
    ])
    
    # Kernel weighting (exponential kernel)
    kernel_width = 0.25 * np.sqrt(n_segments)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    
    print(f"  Distance range: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # Fit weighted linear regression
    print(f"  Fitting weighted linear model...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(masks, predictions, sample_weight=weights)
    
    # Coefficients are importance scores
    importance = ridge.coef_
    
    print(f"  Importance statistics:")
    print(f"    Min: {importance.min():.4f}")
    print(f"    Max: {importance.max():.4f}")
    print(f"    Mean: {importance.mean():.4f}")
    print(f"    Non-zero: {np.count_nonzero(importance)}")
    
    return importance


# ========== VISUALIZATION ==========
def visualize_results(waveform, saliency_map_norm, segment_importance, 
                     filename, output_dir, pred_class, model_type, target_length, sample_rate):
    """Generate and save visualizations."""
    duration = target_length / sample_rate  # Duration in seconds
    
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
    axes[1].set_title(f'Audio with Saliency Heatmap (LIME {model_type.upper()}) - Class {pred_class}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_lime_{model_type}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved visualization: {filename}_lime_{model_type}.png")
    
    # Plot segment importance as bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(range(len(segment_importance)), segment_importance, width=1.0)
    ax.set_title(f'Segment Importance Scores (LIME {model_type.upper()})')
    ax.set_xlabel('Segment Index')
    ax.set_ylabel('Importance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_segment_importance_{model_type}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved segment importance: {filename}_segment_importance_{model_type}.png")


# ========== MAIN SCRIPT ==========
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='LIME Waveform Explanation - Unified for Multiple Models')

    parser.add_argument('--dataset', type=str, default='urbansound8k',
                    choices=['esc50', 'urbansound8k'],
                    help='Dataset to use')
    parser.add_argument('--audio_dir', type=str, default=None,
                    help='Directory containing .wav files (auto-set based on dataset if not specified)')
    parser.add_argument('--model', type=str, choices=['wav2vec2', 'acdnet'], default='wav2vec2',
                       help='Model type to use')
    parser.add_argument('--n_segments', type=int, default=10,
                       help='Number of segments to divide audio into')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of perturbations for LIME')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    args = parser.parse_args()
    
    # Configuration
    model_type = args.model
    dataset = args.dataset
    n_segments = args.n_segments
    num_samples = args.num_samples

    # Get model configuration for the specified dataset (from central config)
    config = get_model_config_for_dataset(model_type, dataset)

    # Set audio directory (auto-determine if not specified)
    if args.audio_dir is None:
        dataset_config = get_dataset_config(dataset)
        audio_dir = dataset_config['default_audio_dir']
    else:
        audio_dir = args.audio_dir
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        output_dir = f"results/saliency/saliency_Lime_{model_type}_{dataset}_{n_segments}_segments"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("LIME WAVEFORM EXPLANATION - UNIFIED")
    print("=" * 80)
    print(f"Dataset: {dataset.upper()}")
    print(f"Model: {model_type.upper()}")
    print(f"Sample rate: {config['sample_rate']} Hz")
    print(f"Target length: {config['target_length']} samples")
    print(f"Duration: {config['target_length'] / config['sample_rate']:.1f}s")
    print(f"Segments: {n_segments}")
    print(f"Perturbations: {num_samples}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load model
    model = load_model(model_type, dataset)
    
    # Get target length
    target_length = config['target_length']
    
    # Get all .wav files from directory
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")
    
    if len(audio_files) == 0:
        print(f"No .wav files found in {audio_dir}")
        sys.exit(1)
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
        print(f"{'='*80}")
        
        # Load audio
        print(f"Loading audio: {audio_file}")
        waveform, orig_sample_rate = torchaudio.load(audio_file)
        
        # Preprocess audio (model-specific)
        waveform = preprocess_audio(waveform, orig_sample_rate, model_type, config)
        print(f"  Final waveform shape: {waveform.shape}")
        
        # Get prediction
        logits, probs = predict_model(model, waveform, model_type)
        pred_class = logits.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
        
        print(f"  Predicted class: {pred_class}, Probability: {pred_prob:.4f}")
        
        # Generate LIME explanation
        print(f"\n  === Starting LIME Waveform ===")
        segment_importance = lime_audio(waveform, model, pred_class, model_type, 
                                       n_segments, num_samples)
        
        # Expand to per-sample saliency map
        segment_size = target_length // n_segments
        saliency_map = np.repeat(segment_importance, segment_size)
        
        # Normalize to [0, 1]
        saliency_map_norm = saliency_map - saliency_map.min()
        if saliency_map_norm.max() > 0:
            saliency_map_norm /= saliency_map_norm.max()
        
        # Save results
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        np.save(os.path.join(output_dir, f'{filename}_lime_{model_type}.npy'), saliency_map_norm)
        print(f"  ✓ Saved saliency map: {filename}_lime_{model_type}.npy")
        
        # Visualization
        visualize_results(waveform, saliency_map_norm, segment_importance,
                         filename, output_dir, pred_class, model_type, target_length, config['sample_rate'])
    
    print("\n" + "="*80)
    print(f"=== LIME Waveform Processing Complete ({model_type.upper()}) ===")
    print(f"Processed {len(audio_files)} files")
    print(f"Results saved to {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()



