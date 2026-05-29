#!/usr/bin/env python
# coding: utf-8

"""
Advanced audio insertion/deletion evaluation with additional features:
- Patch-based masking to reduce artifacts
- Better error handling and logging
- Support for different baseline methods
- Detailed progress tracking
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import glob
import argparse
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom

from src.models.resnet50 import ResNetModel
from src.models.wav2vec2 import Wav2Vec2Model
from src.utils import MODEL_WEIGHTS_PATH, NUM_CLASSES


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedCausalMetric:
    """Advanced causal metric for insertion/deletion evaluation in audio domain."""
    
    def __init__(self, model, mode, n_fft=1024, hop_length=512, 
                 patch_size=(8, 8), baseline_method='zero', substrate_fn=None):
        """
        Initialize advanced causal metric.
        
        Args:
            model: Audio model (ResNet or wav2vec2)
            mode: 'ins' for insertion or 'del' for deletion
            n_fft: FFT window size for STFT
            hop_length: Hop length for STFT
            patch_size: Size of patches for masking (freq, time)
            baseline_method: Method for baseline ('zero', 'mean', 'noise')
            substrate_fn: Function to generate substrate (baseline) audio
        """
        self.model = model
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.baseline_method = baseline_method
        self.substrate_fn = substrate_fn
        
        # Determine model type
        self.is_spectrogram_model = hasattr(model, 'predict') and 'resnet' in str(type(model)).lower()
        self.is_wav2vec2_model = hasattr(model, 'predict') and 'wav2vec' in str(type(model)).lower()
        
        if not (self.is_spectrogram_model or self.is_wav2vec2_model):
            raise ValueError("Model must be either ResNet (spectrogram) or wav2vec2 (raw audio)")
        
        logger.info(f"Initialized {mode} metric for {model_type} model")
    
    def single_run(self, audio_tensor: torch.Tensor, saliency_map: np.ndarray, 
                   steps: int = 50, verbose: int = 0) -> np.ndarray:
        """
        Run single insertion/deletion evaluation with patch-based masking.
        
        Args:
            audio_tensor: Input audio tensor (1, n_samples)
            saliency_map: Saliency map in STFT domain (freq, time)
            steps: Number of steps for evaluation
            verbose: Verbosity level
            
        Returns:
            Array of prediction scores at each step
        """
        # Convert audio to numpy if needed
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.squeeze().cpu().numpy()
        else:
            audio_np = audio_tensor
            
        # Compute STFT
        stft = librosa.stft(audio_np, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Ensure saliency map matches STFT dimensions
        if saliency_map.shape != magnitude.shape:
            logger.warning(f"Resizing saliency map from {saliency_map.shape} to {magnitude.shape}")
            zoom_factors = (magnitude.shape[0] / saliency_map.shape[0], 
                          magnitude.shape[1] / saliency_map.shape[1])
            saliency_map = zoom(saliency_map, zoom_factors, order=1)
        
        # Apply patch-based masking
        if self.patch_size != (1, 1):
            saliency_map = self._apply_patch_averaging(saliency_map, self.patch_size)
        
        # Flatten and get sorted indices
        saliency_flat = saliency_map.flatten()
        if self.mode == 'ins':
            # For insertion: start with baseline, add most important patches first
            sorted_indices = np.argsort(saliency_flat)[::-1]  # Descending order
            current_magnitude = self._get_baseline_magnitude(magnitude)
        else:
            # For deletion: start with original, remove most important patches first
            sorted_indices = np.argsort(saliency_flat)[::-1]  # Descending order
            current_magnitude = magnitude.copy()
        
        # Get original prediction
        original_score = self._get_prediction_score(audio_tensor)
        
        # Calculate step size
        total_patches = len(saliency_flat)
        step_size = max(1, total_patches // steps)
        
        scores = [original_score]
        
        for i in range(1, steps + 1):
            # Calculate number of patches to modify at this step
            n_patches = min(i * step_size, total_patches)
            
            if self.mode == 'ins':
                # Insertion: copy patches from original
                indices_to_modify = sorted_indices[:n_patches]
                current_magnitude = self._apply_patch_modification(
                    current_magnitude, magnitude, indices_to_modify, 'insert'
                )
            else:
                # Deletion: zero out patches
                indices_to_modify = sorted_indices[:n_patches]
                current_magnitude = self._apply_patch_modification(
                    current_magnitude, magnitude, indices_to_modify, 'delete'
                )
            
            # Reconstruct audio
            reconstructed_stft = current_magnitude * np.exp(1j * phase)
            reconstructed_audio = librosa.istft(reconstructed_stft, hop_length=self.hop_length, 
                                              length=len(audio_np))
            
            # Convert back to tensor
            reconstructed_tensor = torch.from_numpy(reconstructed_audio).float().unsqueeze(0)
            
            # Get prediction score
            score = self._get_prediction_score(reconstructed_tensor)
            scores.append(score)
            
            if verbose > 0 and i % (steps // 10) == 0:
                logger.info(f"Step {i}/{steps}: Score = {score:.4f}")
        
        return np.array(scores)
    
    def _apply_patch_averaging(self, saliency_map: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
        """Apply patch-based averaging to saliency map."""
        freq_patch, time_patch = patch_size
        freq_steps = saliency_map.shape[0] // freq_patch
        time_steps = saliency_map.shape[1] // time_patch
        
        # Create patch-averaged saliency map
        patch_saliency = np.zeros((freq_steps, time_steps))
        
        for i in range(freq_steps):
            for j in range(time_steps):
                start_freq = i * freq_patch
                end_freq = min((i + 1) * freq_patch, saliency_map.shape[0])
                start_time = j * time_patch
                end_time = min((j + 1) * time_patch, saliency_map.shape[1])
                
                patch_saliency[i, j] = np.mean(
                    saliency_map[start_freq:end_freq, start_time:end_time]
                )
        
        return patch_saliency
    
    def _get_baseline_magnitude(self, magnitude: np.ndarray) -> np.ndarray:
        """Get baseline magnitude based on method."""
        if self.baseline_method == 'zero':
            return np.zeros_like(magnitude)
        elif self.baseline_method == 'mean':
            return np.full_like(magnitude, np.mean(magnitude))
        elif self.baseline_method == 'noise':
            return np.random.normal(0, np.std(magnitude), magnitude.shape)
        else:
            return np.zeros_like(magnitude)
    
    def _apply_patch_modification(self, current_magnitude: np.ndarray, 
                                 original_magnitude: np.ndarray, 
                                 patch_indices: np.ndarray, 
                                 operation: str) -> np.ndarray:
        """Apply patch-based modification to magnitude."""
        freq_patch, time_patch = self.patch_size
        freq_steps = current_magnitude.shape[0] // freq_patch
        time_steps = current_magnitude.shape[1] // time_patch
        
        result = current_magnitude.copy()
        
        for patch_idx in patch_indices:
            i = patch_idx // time_steps
            j = patch_idx % time_steps
            
            start_freq = i * freq_patch
            end_freq = min((i + 1) * freq_patch, current_magnitude.shape[0])
            start_time = j * time_patch
            end_time = min((j + 1) * time_patch, current_magnitude.shape[1])
            
            if operation == 'insert':
                result[start_freq:end_freq, start_time:end_time] = \
                    original_magnitude[start_freq:end_freq, start_time:end_time]
            elif operation == 'delete':
                result[start_freq:end_freq, start_time:end_time] = 0.0
        
        return result
    
    def _get_prediction_score(self, audio_tensor: torch.Tensor) -> float:
        """Get prediction score for the target class."""
        with torch.no_grad():
            if self.is_spectrogram_model:
                score = self._get_spectrogram_prediction(audio_tensor)
            else:
                score = self._get_raw_audio_prediction(audio_tensor)
        
        return score
    
    def _get_spectrogram_prediction(self, audio_tensor: torch.Tensor) -> float:
        """Get prediction for spectrogram-based model."""
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )
        
        mel_spec = mel_transform(audio_tensor)
        S_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Convert to PIL image
        S_np = S_db.squeeze().numpy()
        S_norm = (S_np - S_np.min()) / (S_np.max() - S_np.min() + 1e-6)
        S_img = (S_norm * 255).astype(np.uint8)
        
        img = Image.fromarray(S_img)
        img = img.convert("L")
        img = img.resize((224, 224), Image.BILINEAR)
        
        # Convert to RGB and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Get prediction
        logits, probs = self.model.predict(img_tensor)
        return torch.max(probs).item()
    
    def _get_raw_audio_prediction(self, audio_tensor: torch.Tensor) -> float:
        """Get prediction for raw audio model (wav2vec2)."""
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
        
        # Get prediction
        logits, probs = self.model.predict(audio_tensor)
        return torch.max(probs).item()


def auc(scores: np.ndarray) -> float:
    """Calculate Area Under Curve using trapezoidal rule."""
    x = np.linspace(0, 1, len(scores))
    return np.trapz(scores, x)


def load_audio_file(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and return tensor and sample rate."""
    try:
        audio, sr = torchaudio.load(file_path, channels_first=True)
        
        # Convert to mono if multi-channel
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def find_audio_files(audio_dir: str) -> List[str]:
    """Find all audio files in directory."""
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return sorted(audio_files)


def evaluate_audio_saliency_maps_advanced(
    audio_dir: str,
    saliency_dir: str,
    output_dir: str,
    model_type: str = 'resnet',
    model_path: str = None,
    file_suffix: str = '_saliency.npy',
    n_fft: int = 1024,
    hop_length: int = 512,
    steps: int = 50,
    patch_size: Tuple[int, int] = (8, 8),
    baseline_method: str = 'zero'
):
    """
    Advanced evaluation of audio saliency maps using insertion/deletion metrics.
    
    Args:
        audio_dir: Directory containing audio files
        saliency_dir: Directory containing saliency maps
        output_dir: Output directory for results
        model_type: Type of model ('resnet' or 'wav2vec2')
        model_path: Path to model weights
        file_suffix: Suffix for saliency map files
        n_fft: FFT window size
        hop_length: Hop length for STFT
        steps: Number of evaluation steps
        patch_size: Size of patches for masking
        baseline_method: Method for baseline generation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting advanced audio saliency evaluation")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Saliency directory: {saliency_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Patch size: {patch_size}")
    logger.info(f"Baseline method: {baseline_method}")
    
    # Load model
    logger.info(f"Loading {model_type} model...")
    try:
        if model_type.lower() == 'resnet':
            model = ResNetModel(weights_path=model_path)
        elif model_type.lower() == 'wav2vec2':
            model = Wav2Vec2Model(weights_path=model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Initialize metrics
    logger.info("Initializing evaluation metrics...")
    insertion_metric = AdvancedCausalMetric(
        model, 'ins', n_fft, hop_length, patch_size, baseline_method
    )
    deletion_metric = AdvancedCausalMetric(
        model, 'del', n_fft, hop_length, patch_size, baseline_method
    )
    
    # Find audio files
    audio_files = find_audio_files(audio_dir)
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logger.error("No audio files found")
        return
    
    # Results storage
    insertion_scores = []
    deletion_scores = []
    audio_names = []
    detailed_results = []
    
    logger.info("Starting evaluation...")
    for audio_path in tqdm(audio_files, desc="Evaluating audio files"):
        try:
            # Load audio
            audio_tensor, sr = load_audio_file(audio_path)
            
            # Get corresponding saliency map
            rel_path = os.path.relpath(audio_path, audio_dir)
            name_without_ext = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Look for saliency map
            saliency_path = os.path.join(
                saliency_dir,
                rel_path.replace(os.path.splitext(rel_path)[1], file_suffix)
            )
            
            if not os.path.exists(saliency_path):
                # Try without subdirectory
                saliency_path = os.path.join(saliency_dir, f'{name_without_ext}{file_suffix}')
            
            if not os.path.exists(saliency_path):
                logger.warning(f"Saliency map not found for {audio_path}")
                continue
            
            # Load saliency map
            saliency_map = np.load(saliency_path)
            
            # Ensure saliency map is 2D (freq, time)
            if saliency_map.ndim != 2:
                logger.warning(f"Unexpected saliency map shape {saliency_map.shape} for {audio_path}")
                continue
            
            # Run insertion evaluation
            insertion_score = insertion_metric.single_run(
                audio_tensor, saliency_map, steps=steps, verbose=0
            )
            insertion_auc_score = auc(insertion_score)
            
            # Run deletion evaluation
            deletion_score = deletion_metric.single_run(
                audio_tensor, saliency_map, steps=steps, verbose=0
            )
            deletion_auc_score = auc(deletion_score)
            
            # Store results
            insertion_scores.append(insertion_auc_score)
            deletion_scores.append(deletion_auc_score)
            audio_names.append(name_without_ext)
            
            detailed_results.append({
                'audio_name': name_without_ext,
                'insertion_auc': insertion_auc_score,
                'deletion_auc': deletion_auc_score,
                'insertion_curve': insertion_score,
                'deletion_curve': deletion_score
            })
            
            logger.info(f"{name_without_ext}: Insertion AUC = {insertion_auc_score:.4f}, "
                       f"Deletion AUC = {deletion_auc_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    # Calculate mean AUCs
    if insertion_scores and deletion_scores:
        mean_insertion_auc = np.mean(insertion_scores)
        mean_deletion_auc = np.mean(deletion_scores)
        std_insertion_auc = np.std(insertion_scores)
        std_deletion_auc = np.std(deletion_scores)
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Mean Insertion AUC: {mean_insertion_auc:.4f} ± {std_insertion_auc:.4f}")
        logger.info(f"Mean Deletion AUC: {mean_deletion_auc:.4f} ± {std_deletion_auc:.4f}")
        logger.info(f"Number of audio files evaluated: {len(insertion_scores)}")
        
        # Save detailed results
        results = {
            'audio_names': audio_names,
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'mean_insertion_auc': mean_insertion_auc,
            'mean_deletion_auc': mean_deletion_auc,
            'std_insertion_auc': std_insertion_auc,
            'std_deletion_auc': std_deletion_auc,
            'total_audio_files': len(insertion_scores),
            'detailed_results': detailed_results,
            'evaluation_params': {
                'model_type': model_type,
                'n_fft': n_fft,
                'hop_length': hop_length,
                'steps': steps,
                'patch_size': patch_size,
                'baseline_method': baseline_method
            }
        }
        
        np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
        
        # Save as text file
        with open(os.path.join(output_dir, 'evaluation_summary_rise.txt'), 'w') as f:
            f.write("=== ADVANCED AUDIO SALIENCY MAP EVALUATION ===\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Patch Size: {patch_size}\n")
            f.write(f"Baseline Method: {baseline_method}\n")
            f.write(f"Mean Insertion AUC: {mean_insertion_auc:.4f} ± {std_insertion_auc:.4f}\n")
            f.write(f"Mean Deletion AUC: {mean_deletion_auc:.4f} ± {std_deletion_auc:.4f}\n")
            f.write(f"Number of audio files evaluated: {len(insertion_scores)}\n\n")
            f.write("Detailed Results:\n")
            f.write("Audio Name\tInsertion AUC\tDeletion AUC\n")
            f.write("-" * 50 + "\n")
            for name, ins_auc, del_auc in zip(audio_names, insertion_scores, deletion_scores):
                f.write(f"{name}\t{ins_auc:.4f}\t{del_auc:.4f}\n")
        
        # Create visualization
        create_evaluation_plots(insertion_scores, deletion_scores, 
                               mean_insertion_auc, mean_deletion_auc, output_dir)
        
        logger.info(f"Results saved to {output_dir}/")
        
    else:
        logger.error("No valid audio files found for evaluation.")


def create_evaluation_plots(insertion_scores: List[float], deletion_scores: List[float],
                           mean_insertion_auc: float, mean_deletion_auc: float,
                           output_dir: str):
    """Create evaluation plots."""
    plt.figure(figsize=(15, 5))
    
    # AUC distributions
    plt.subplot(131)
    plt.hist(insertion_scores, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mean_insertion_auc, color='red', linestyle='--', 
                label=f'Mean: {mean_insertion_auc:.4f}')
    plt.xlabel('Insertion AUC')
    plt.ylabel('Frequency')
    plt.title('Insertion AUC Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(132)
    plt.hist(deletion_scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(mean_deletion_auc, color='red', linestyle='--', 
                label=f'Mean: {mean_deletion_auc:.4f}')
    plt.xlabel('Deletion AUC')
    plt.ylabel('Frequency')
    plt.title('Deletion AUC Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(133)
    plt.scatter(insertion_scores, deletion_scores, alpha=0.6, s=50)
    plt.xlabel('Insertion AUC')
    plt.ylabel('Deletion AUC')
    plt.title('Insertion vs Deletion AUC')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    min_val = min(min(insertion_scores), min(deletion_scores))
    max_val = max(max(insertion_scores), max(deletion_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Advanced audio saliency map evaluation')
    parser.add_argument('--audio_dir', required=True, help='Directory containing audio files')
    parser.add_argument('--saliency_dir', required=True, help='Directory containing saliency maps')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--model_type', default='resnet', choices=['resnet', 'wav2vec2'], 
                       help='Type of model to use')
    parser.add_argument('--model_path', default=None, help='Path to model weights')
    parser.add_argument('--file_suffix', default='_saliency.npy', help='Suffix for saliency map files')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')
    parser.add_argument('--steps', type=int, default=50, help='Number of evaluation steps')
    parser.add_argument('--patch_size', nargs=2, type=int, default=[8, 8], 
                       help='Patch size for masking (freq, time)')
    parser.add_argument('--baseline_method', default='zero', 
                       choices=['zero', 'mean', 'noise'],
                       help='Baseline method for insertion')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        if args.model_type == 'resnet':
            args.model_path = MODEL_WEIGHTS_PATH
        else:
            args.model_path = 'wav2vec2_esc50.pt'
    
    evaluate_audio_saliency_maps_advanced(
        audio_dir=args.audio_dir,
        saliency_dir=args.saliency_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        model_path=args.model_path,
        file_suffix=args.file_suffix,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        steps=args.steps,
        patch_size=tuple(args.patch_size),
        baseline_method=args.baseline_method
    )


if __name__ == "__main__":
    main()

