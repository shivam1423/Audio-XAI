#!/usr/bin/env python
# coding: utf-8
"""
Evaluation Script for Audio Saliency Maps - UNIFIED
---------------------------------------------------
Computes Insertion and Deletion AUC for audio explanations
Supports: Wav2Vec2, ACDNet

Usage:
    python evaluate_insertion_deletion_waveform.py \
        --model wav2vec2 \
        --audio <audio_dir> \
        --maps_dir <saliency_dir> \
        --output_dir <results_dir>
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
import os
import glob
import argparse
import sys

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
    wav2vec2_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Wav2Vec2'))
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
    
    # Wrap with softmax for probability output
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("✓ Wav2Vec2 model loaded")
    return model


def load_acdnet_model(checkpoint_path, num_classes, sample_rate, target_length):
    """Load ACDNet model."""
    # Get absolute path to ACDNet directory
    # sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ACDNet/torch'))
    # from resources.models import GetACDNetModel
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


def predict_model(model, waveform, model_type):
    """
    Get prediction from model.
    
    Parameters
    ----------
    model : nn.Module
        Model instance
    waveform : torch.Tensor
        Preprocessed waveform (target_length,) or (1, target_length)
    model_type : str
        Model type
    
    Returns
    -------
    probs : torch.Tensor
        Model probabilities (1, num_classes)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if model_type == 'wav2vec2':
        # Wav2Vec2: expects (batch, samples)
        # Already has softmax wrapper
        with torch.no_grad():
            probs = model(waveform.to(device))
    
    elif model_type == 'acdnet':
        # ACDNet: expects (batch, 1, 1, samples) for Conv2D
        input_tensor = waveform.unsqueeze(1).unsqueeze(1).to(device)
        
        with torch.no_grad():
            probs = model(input_tensor)  # ACDNet has softmax already
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return probs


# ========== EVALUATION METRICS ==========
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():
    """
    Insertion/Deletion metric for audio explanations
    Adapted for multiple models
    """

    def __init__(self, model, model_type, mode, step, substrate_fn):
        """
        Args:
            model (nn.Module): Black-box model being explained
            model_type (str): 'wav2vec2' or 'acdnet'
            mode (str): 'del' or 'ins'
            step (int): number of samples modified per iteration
            substrate_fn (func): mapping from old samples to new samples
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.model_type = model_type
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, audio_tensor, explanation, verbose=0):
        """
        Run metric on one audio-saliency pair

        Args:
            audio_tensor (Tensor): audio waveform (1, target_length) or (target_length,)
            explanation (np.ndarray): saliency map (target_length,)
            verbose (int): 0, 1, or 2 for different levels of output

        Returns:
            scores (np.ndarray): Array containing scores at every step
        """
        # Ensure audio has batch dimension
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Get prediction for original audio using unified interface
        pred = predict_model(self.model, audio_tensor, self.model_type)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]

        total_samples = audio_tensor.shape[-1]
        n_steps = (total_samples + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Samples deleted'
            start = audio_tensor.clone()
            finish = self.substrate_fn(audio_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Samples inserted'
            start = self.substrate_fn(audio_tensor)
            finish = audio_tensor.clone()

        scores = np.empty(n_steps + 1)

        # Sort samples by saliency (most important first)
        salient_order = np.flip(np.argsort(explanation.reshape(-1), axis=-1), axis=-1)

        for i in range(n_steps + 1):
            pred = predict_model(self.model, start, self.model_type)
            pr, cl = torch.topk(pred, 2)

            if verbose == 2:
                print(f'Step {i}: Class {cl[0][0].item()}: {float(pr[0][0]):.3f}')
                print(f'        Class {cl[0][1].item()}: {float(pr[0][1]):.3f}')

            scores[i] = pred[0, c].cpu().item()

            # Visualization (if requested)
            if verbose >= 1 and (i == n_steps or verbose == 2):
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title(f'{ylabel} {100 * i / n_steps:.1f}%, P={scores[i]:.4f}')
                plt.plot(start[0].cpu().numpy())
                plt.xlabel('Sample')
                plt.ylabel('Amplitude')

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(f'Class {c}')
                plt.show()

            # Modify audio for next iteration
            if i < n_steps:
                coords = salient_order[self.step * i:self.step * (i + 1)]
                start_flat = start.cpu().numpy().reshape(1, -1)
                finish_flat = finish.cpu().numpy().reshape(1, -1)
                start_flat[0, coords] = finish_flat[0, coords]
                start = torch.from_numpy(start_flat.reshape(start.shape)).float()

        return scores


# ========== MAIN SCRIPT ==========
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate Audio Saliency Maps - Unified for Multiple Models')
    parser.add_argument('--dataset', type=str, choices=['esc50', 'urbansound8k'], default='urbansound8k',
                       help='Dataset to use (default: urbansound8k)')
    parser.add_argument('--model', type=str, choices=['wav2vec2', 'acdnet'], default='wav2vec2',
                       help='Model type to use')
    parser.add_argument('--method', type=str, choices=['lime', 'rise', 'rise_mo'], default='rise',
                       help='Saliency method that produced the maps: lime, rise, or rise_mo (default: rise)')
    parser.add_argument('--audio', required=True, help='Directory containing audio files')
    parser.add_argument('--maps_dir', required=True, help='Directory containing saliency maps')
    parser.add_argument('--suffix', default=None,
                       help='Saliency map file suffix override (default: auto-derived from --method and --model)')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    args = parser.parse_args()

    model_type = args.model
    dataset = args.dataset
    audio_dir = args.audio
    saliency_maps_dir = args.maps_dir
    output_dir = args.output_dir

    # Get configuration from central config
    config = get_model_config_for_dataset(model_type, dataset)
    target_sample_rate = config['sample_rate']
    target_length = config['target_length']

    # Auto-derive saliency file suffix from method + model if not explicitly provided
    # lime → _lime_{model}.npy  |  rise / rise_mo → _rise_{model}.npy
    method_tag = 'lime' if args.method == 'lime' else 'rise'
    file_suffix = args.suffix if args.suffix else f'_{method_tag}_{model_type}.npy'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("AUDIO SALIENCY MAP EVALUATION - UNIFIED")
    print("=" * 80)
    print(f"Dataset: {dataset.upper()}")
    print(f"Model: {model_type.upper()}")
    print(f"Method: {args.method.upper()}")
    print(f"Sample rate: {target_sample_rate} Hz")
    print(f"Target length: {target_length} samples")
    print(f"Duration: {target_length / target_sample_rate:.1f}s")
    print(f"File suffix: {file_suffix}")
    print(f"Audio directory: {audio_dir}")
    print(f"Saliency maps directory: {saliency_maps_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load model
    model = load_model(model_type, dataset)
    
    # Define substrate functions
    silence_fn = lambda x: torch.zeros_like(x)
    
    # Initialize metrics
    print("Initializing evaluation metrics...")
    step_size = target_length // 224  # 224 evaluation points
    insertion_metric = CausalMetric(model, model_type, 'ins', step_size, substrate_fn=silence_fn)
    deletion_metric = CausalMetric(model, model_type, 'del', step_size, substrate_fn=silence_fn)
    print(f"Step size: {step_size} samples per step")
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
    print(f"Found {len(audio_files)} audio files")
    
    # Results storage
    insertion_scores = []
    deletion_scores = []
    audio_names = []
    
    print("\nStarting evaluation...")
    for audio_path in tqdm(audio_files, desc="Evaluating audio files"):
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)  # Average channels
            else:
                waveform = waveform.squeeze(0)  # Remove channel dimension  # Remove channel dimension
            
            # Pad or trim to target length
            if waveform.shape[0] < target_length:
                # Pad with zeros
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[0]))
            elif waveform.shape[0] > target_length:
                # For ACDNet, use center crop; for Wav2Vec2, use regular crop
                if model_type == 'acdnet':
                    start = (waveform.shape[0] - target_length) // 2
                    waveform = waveform[start:start + target_length]
                else:
                    waveform = waveform[:target_length]
            
            # Get corresponding saliency map
            audio_basename = os.path.basename(audio_path)
            name_without_ext = os.path.splitext(audio_basename)[0]
            
            # Look for saliency map
            saliency_path = os.path.join(saliency_maps_dir, f'{name_without_ext}{file_suffix}')
            
            if not os.path.exists(saliency_path):
                print(f"Warning: Saliency map not found for {audio_path}")
                continue
            
            # Load saliency map
            saliency_map = np.load(saliency_path)
            
            # Verify shape
            if saliency_map.shape[0] != target_length:
                print(f"Warning: Saliency map shape {saliency_map.shape} doesn't match audio length {target_length}")
                continue
            
            # Run insertion evaluation
            insertion_score = insertion_metric.single_run(waveform, saliency_map, verbose=0)
            insertion_auc_val = auc(insertion_score)
            
            # Run deletion evaluation
            deletion_score = deletion_metric.single_run(waveform, saliency_map, verbose=0)
            deletion_auc_val = auc(deletion_score)
            
            # Store results
            insertion_scores.append(insertion_auc_val)
            deletion_scores.append(deletion_auc_val)
            audio_names.append(name_without_ext)
            
            print(f"{name_without_ext}: Insertion AUC = {insertion_auc_val:.4f}, Deletion AUC = {deletion_auc_val:.4f}")
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate mean AUCs
    if insertion_scores and deletion_scores:
        mean_insertion_auc = np.mean(insertion_scores)
        mean_deletion_auc = np.mean(deletion_scores)
        
        print(f"\n=== EVALUATION RESULTS ({model_type.upper()}) ===")
        print(f"Mean Insertion AUC: {mean_insertion_auc:.4f}")
        print(f"Mean Deletion AUC: {mean_deletion_auc:.4f}")
        print(f"Number of audio files evaluated: {len(insertion_scores)}")
        
        # Save detailed results
        results = {
            'model_type': model_type,
            'audio_names': audio_names,
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'mean_insertion_auc': mean_insertion_auc,
            'mean_deletion_auc': mean_deletion_auc,
            'total_audio': len(insertion_scores)
        }
        
        np.save(os.path.join(output_dir, f'evaluation_results_{model_type}.npy'), results)
        
        # Save as text file
        with open(os.path.join(output_dir, f'evaluation_summary_{model_type}.txt'), 'w') as f:
            f.write(f"=== AUDIO SALIENCY MAP EVALUATION ({model_type.upper()}) ===\n\n")
            f.write(f"Mean Insertion AUC: {mean_insertion_auc:.4f}\n")
            f.write(f"Mean Deletion AUC: {mean_deletion_auc:.4f}\n")
            f.write(f"Number of audio files evaluated: {len(insertion_scores)}\n\n")
            f.write("Detailed Results:\n")
            f.write("Audio Name\tInsertion AUC\tDeletion AUC\n")
            f.write("-" * 60 + "\n")
            for name, ins_auc, del_auc in zip(audio_names, insertion_scores, deletion_scores):
                f.write(f"{name}\t{ins_auc:.4f}\t{del_auc:.4f}\n")
        
        print(f"\nResults saved to {output_dir}/")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.hist(insertion_scores, bins=10, alpha=0.7, color='blue')
        plt.axvline(mean_insertion_auc, color='red', linestyle='--',
                    label=f'Mean: {mean_insertion_auc:.4f}')
        plt.xlabel('Insertion AUC')
        plt.ylabel('Frequency')
        plt.title(f'Insertion AUC Distribution ({model_type.upper()})')
        plt.legend()
        
        plt.subplot(122)
        plt.hist(deletion_scores, bins=10, alpha=0.7, color='green')
        plt.axvline(mean_deletion_auc, color='red', linestyle='--',
                    label=f'Mean: {mean_deletion_auc:.4f}')
        plt.xlabel('Deletion AUC')
        plt.ylabel('Frequency')
        plt.title(f'Deletion AUC Distribution ({model_type.upper()})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'auc_distributions_{model_type}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Evaluation completed successfully!")
    
    else:
        print("No valid audio files found for evaluation.")


if __name__ == '__main__':
    main()