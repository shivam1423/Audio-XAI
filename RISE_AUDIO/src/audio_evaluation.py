#!/usr/bin/env python
# coding: utf-8

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
from typing import Tuple, List, Optional, Union
from PIL import Image
import torchvision.transforms as transforms

from src.models.resnet50 import ResNetModel
from src.models.wav2vec2 import Wav2Vec2Model
from src.utils import MODEL_WEIGHTS_PATH, NUM_CLASSES


class CausalMetric:
    """Causal metric for insertion/deletion evaluation in audio domain."""
    
    def __init__(self, model, mode, n_fft=1024, hop_length=512, substrate_fn=None):
        """
        Initialize causal metric.
        
        Args:
            model: Audio model (ResNet or wav2vec2)
            mode: 'ins' for insertion or 'del' for deletion
            n_fft: FFT window size for STFT
            hop_length: Hop length for STFT
            substrate_fn: Function to generate substrate (baseline) audio
        """
        self.model = model
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.substrate_fn = substrate_fn
        
        # Determine model type
        self.is_spectrogram_model = hasattr(model, 'predict') and 'resnet' in str(type(model)).lower()
        self.is_wav2vec2_model = hasattr(model, 'predict') and 'wav2vec' in str(type(model)).lower()
        
        if not (self.is_spectrogram_model or self.is_wav2vec2_model):
            raise ValueError("Model must be either ResNet (spectrogram) or wav2vec2 (raw audio)")
    
    def single_run(self, audio_tensor: torch.Tensor, saliency_map: np.ndarray, 
                   steps: int = 50, verbose: int = 0, audio_sr: Optional[int] = None) -> np.ndarray:
        """
        Run single insertion/deletion evaluation.
        
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
            # Resize saliency map to match STFT dimensions
            print(f"Warning: Saliency shape {saliency_map.shape} not matches with spectrogram shape {magnitude.shape}")
            from scipy.ndimage import zoom
            zoom_factors = (magnitude.shape[0] / saliency_map.shape[0], 
                          magnitude.shape[1] / saliency_map.shape[1])
            saliency_map = zoom(saliency_map, zoom_factors, order=1)
        
        # Flatten and get sorted indices
        saliency_flat = saliency_map.flatten()
        if self.mode == 'ins':
            # For insertion: start with zeros, add most important bins first
            sorted_indices = np.argsort(saliency_flat)[::-1]  # Descending order
            current_magnitude = np.zeros_like(magnitude)
        else:
            # For deletion: start with original, remove most important bins first
            sorted_indices = np.argsort(saliency_flat)[::-1]  # Descending order
            current_magnitude = magnitude.copy()
        
        # Get original prediction
        original_score = self._get_prediction_score(audio_tensor, audio_sr)
        
        # Calculate step size
        total_bins = len(saliency_flat)
        step_size = max(1, total_bins // steps)
        
        scores = [original_score]
        
        for i in range(1, steps + 1):
            # Calculate number of bins to modify at this step
            n_bins = min(i * step_size, total_bins)
            
            if self.mode == 'ins':
                # Insertion: copy bins from original
                indices_to_modify = sorted_indices[:n_bins]
                current_magnitude.flat[indices_to_modify] = magnitude.flat[indices_to_modify]
            else:
                # Deletion: zero out bins
                indices_to_modify = sorted_indices[:n_bins]
                current_magnitude.flat[indices_to_modify] = 0.0
            
            # Reconstruct audio
            reconstructed_stft = current_magnitude * np.exp(1j * phase)
            reconstructed_audio = librosa.istft(reconstructed_stft, hop_length=self.hop_length, 
                                              length=len(audio_np))
            
            # Convert back to tensor
            reconstructed_tensor = torch.from_numpy(reconstructed_audio).float().unsqueeze(0)
            
            # Get prediction score
            score = self._get_prediction_score(reconstructed_tensor, audio_sr)
            scores.append(score)
            
            if verbose > 0 and i % (steps // 10) == 0:
                print(f"Step {i}/{steps}: Score = {score:.4f}")
        
        return np.array(scores)
    
    def _get_prediction_score(self, audio_tensor: torch.Tensor, audio_sr: Optional[int]) -> float:
        """Get prediction score for the target class."""
        with torch.no_grad():
            if self.is_spectrogram_model:
                # For ResNet: convert to mel spectrogram first
                score = self._get_spectrogram_prediction(audio_tensor, audio_sr)
            else:
                # For wav2vec2: use raw audio
                score = self._get_raw_audio_prediction(audio_tensor)
        
        return score
    
    def _get_spectrogram_prediction(self, audio_tensor: torch.Tensor, audio_sr: Optional[int]) -> float:
        """Get prediction for spectrogram-based model."""
        # Resample to 22050 Hz for consistent mel parameters
        target_sr = 22050
        x = audio_tensor
        if audio_sr is not None and audio_sr != target_sr:
            x = torchaudio.functional.resample(x, orig_freq=audio_sr, new_freq=target_sr)

        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )
        
        mel_spec = mel_transform(x)
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
    audio, sr = torchaudio.load(file_path, channels_first=True)
    
    # Convert to mono if multi-channel
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    return audio, sr


def find_audio_files(audio_dir: str) -> List[str]:
    """Find all audio files in directory."""
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return audio_files


def evaluate_audio_saliency_maps(
    audio_dir: str,
    saliency_dir: str,
    output_dir: str,
    model_type: str = 'resnet',
    model_path: str = None,
    file_suffix: str = '_saliency.npy',
    n_fft: int = 1024,
    hop_length: int = 512,
    steps: int = 50
):
    """
    Evaluate audio saliency maps using insertion/deletion metrics.
    
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
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading {model_type} model...")
    if model_type.lower() == 'resnet':
        model = ResNetModel(weights_path=model_path)
    elif model_type.lower() == 'wav2vec2':
        model = Wav2Vec2Model(weights_path=model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize metrics
    print("Initializing evaluation metrics...")
    insertion_metric = CausalMetric(model, 'ins', n_fft, hop_length)
    deletion_metric = CausalMetric(model, 'del', n_fft, hop_length)
    
    # Find audio files
    audio_files = find_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    # Results storage
    insertion_scores = []
    deletion_scores = []
    audio_names = []
    
    print("Starting evaluation...")
    for audio_path in tqdm(audio_files, desc="Evaluating audio files"):
        try:
            # Load audio
            audio_tensor, sr = load_audio_file(audio_path)
            
            # Get corresponding saliency map
            rel_path = os.path.relpath(audio_path, audio_dir)
            name_without_ext = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Look for saliency map with multiple naming patterns
            cand_rel = os.path.join(
                saliency_dir,
                rel_path.replace(os.path.splitext(rel_path)[1], file_suffix)
            )
            cand_flat = os.path.join(saliency_dir, f'{name_without_ext}{file_suffix}')
            # Generator pattern: saliency_<basename>.wav_saliency.npy
            basename_with_ext = os.path.basename(audio_path)
            cand_generator = os.path.join(saliency_dir, f'saliency_{basename_with_ext}_saliency.npy')
            # Another generator variant given file_suffix already includes .wav_saliency.npy
            cand_generator_suffix = os.path.join(saliency_dir, f'saliency_{basename_with_ext.replace(os.path.splitext(basename_with_ext)[1], file_suffix)}')

            for candidate in [cand_rel, cand_flat, cand_generator, cand_generator_suffix]:
                if os.path.exists(candidate):
                    saliency_path = candidate
                    break
            else:
                print(f"Warning: Saliency map not found for {audio_path}")
                continue
            
            # Load saliency map
            saliency_map = np.load(saliency_path)
            
            # Ensure saliency map is 2D (freq, time)
            if saliency_map.ndim != 2:
                print(f"Warning: Unexpected saliency map shape {saliency_map.shape} for {audio_path}")
                continue
            
            # Run insertion evaluation
            insertion_score = insertion_metric.single_run(audio_tensor, saliency_map, steps=steps, verbose=0, audio_sr=sr)
            insertion_auc_score = auc(insertion_score)
            
            # Run deletion evaluation
            deletion_score = deletion_metric.single_run(audio_tensor, saliency_map, steps=steps, verbose=0, audio_sr=sr)
            deletion_auc_score = auc(deletion_score)
            
            # Store results
            insertion_scores.append(insertion_auc_score)
            deletion_scores.append(deletion_auc_score)
            audio_names.append(name_without_ext)
            
            print(f"{name_without_ext}: Insertion AUC = {insertion_auc_score:.4f}, Deletion AUC = {deletion_auc_score:.4f}")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Calculate mean AUCs
    if insertion_scores and deletion_scores:
        mean_insertion_auc = np.mean(insertion_scores)
        mean_deletion_auc = np.mean(deletion_scores)
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Mean Insertion AUC: {mean_insertion_auc:.4f}")
        print(f"Mean Deletion AUC: {mean_deletion_auc:.4f}")
        print(f"Number of audio files evaluated: {len(insertion_scores)}")
        
        # Save detailed results
        results = {
            'audio_names': audio_names,
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'mean_insertion_auc': mean_insertion_auc,
            'mean_deletion_auc': mean_deletion_auc,
            'total_audio_files': len(insertion_scores)
        }
        
        np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
        
        # Save as text file
        with open(os.path.join(output_dir, 'evaluation_summary_rise.txt'), 'w') as f:
            f.write("=== AUDIO SALIENCY MAP EVALUATION ===\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Mean Insertion AUC: {mean_insertion_auc:.4f}\n")
            f.write(f"Mean Deletion AUC: {mean_deletion_auc:.4f}\n")
            f.write(f"Number of audio files evaluated: {len(insertion_scores)}\n\n")
            f.write("Detailed Results:\n")
            f.write("Audio Name\tInsertion AUC\tDeletion AUC\n")
            f.write("-" * 50 + "\n")
            for name, ins_auc, del_auc in zip(audio_names, insertion_scores, deletion_scores):
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
        print("No valid audio files found for evaluation.")


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio saliency maps using insertion/deletion metrics')
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
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        if args.model_type == 'resnet':
            args.model_path = MODEL_WEIGHTS_PATH
        else:
            args.model_path = 'wav2vec2_esc50.pt'  # Update with actual path
    
    evaluate_audio_saliency_maps(
        audio_dir=args.audio_dir,
        saliency_dir=args.saliency_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        model_path=args.model_path,
        file_suffix=args.file_suffix,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        steps=args.steps
    )


if __name__ == "__main__":
    main()

