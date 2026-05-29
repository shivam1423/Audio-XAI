#!/usr/bin/env python
# coding: utf-8

"""Audio preprocessing for spectrogram-based models (ResNet50, HTSAT)."""

import torch
try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None
import librosa
import numpy as np
from PIL import Image
from typing import Union, Optional


class SpectrogramPreprocessor:
    """
    Preprocessor for converting raw audio to spectrograms for image-based models.
    Supports both ResNet50 (224x224 RGB images) and HTSAT (mel spectrograms).
    """
    
    def __init__(
        self,
        model_type: str,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float = 0,
        fmax: Optional[float] = None,
        input_size: tuple = (224, 224),
        normalize_mean: list = [0.5, 0.5, 0.5],
        normalize_std: list = [0.5, 0.5, 0.5],
        clip_samples: Optional[int] = None
    ):
        """
        Initialize spectrogram preprocessor.
        
        Args:
            model_type: Model type ('resnet50' or 'htsat')
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel bins
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sr/2)
            input_size: Target image size (H, W)
            normalize_mean: Mean for normalization
            normalize_std: Std for normalization
            clip_samples: Max audio length in samples (for padding/truncating)
        """
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.input_size = input_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.clip_samples = clip_samples
        
    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load audio file and convert to mono at target sample rate.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            if torchaudio is None:
                raise RuntimeError("torchaudio not available")
            waveform, sr = torchaudio.load(filepath)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # Convert to numpy
            audio_np = waveform.squeeze(0).numpy()
            
        except Exception as e:
            print(f"torchaudio failed ({e}), trying librosa...")
            # Fallback to librosa
            audio_np, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        
        # Pad or truncate if clip_samples is specified
        if self.clip_samples is not None:
            if len(audio_np) < self.clip_samples:
                # Zero-pad to target length (matches Resnet50_UrbanSound8K training)
                audio_np = np.pad(audio_np, (0, self.clip_samples - len(audio_np)), mode='constant')
            else:
                # Truncate to exact length
                audio_np = audio_np[:self.clip_samples]
        
        return audio_np
    
    def audio_to_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel-spectrogram using librosa.
        This exactly replicates the original preprocessing pipeline.
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            Mel-spectrogram in dB scale
        """
        # Generate mel spectrogram (matching original parameters)
        S = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to dB scale (matching original)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        return S_db
    
    def spectrogram_to_image(self, mel_spec: np.ndarray) -> torch.Tensor:
        """
        Convert mel-spectrogram to normalized image tensor.
        For ResNet50: Returns 3-channel RGB tensor
        For HTSAT: Returns single-channel tensor (HTSAT handles rest internally)
        
        Args:
            mel_spec: Mel-spectrogram in dB scale
            
        Returns:
            Normalized image tensor
        """
        # Normalize to [0, 1] (matching original [0,255] then /255)
        S_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        
        if self.model_type == 'resnet50':
            # Convert to PIL Image and resize (matching original)
            S_img = (S_norm * 255).astype(np.uint8)
            img = Image.fromarray(S_img)
            
            # Determine if grayscale or RGB based on normalize_mean length
            num_channels = len(self.normalize_mean)
            if num_channels == 1:
                # Grayscale (1 channel) - matching training
                img = img.convert("L")
                img = img.resize(self.input_size, Image.BILINEAR)
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
                # Apply normalization
                mean = torch.tensor(self.normalize_mean).view(1, 1, 1)
                std = torch.tensor(self.normalize_std).view(1, 1, 1)
                img_tensor = (img_tensor - mean) / std
            else:
                # RGB (3 channels) - replicate grayscale
                img = img.convert("L")
                img = img.resize(self.input_size, Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
                img_tensor = img_tensor.repeat(3, 1, 1)  # Replicate to 3 channels
                # Apply normalization
                mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
                std = torch.tensor(self.normalize_std).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
            # else:
            #     # RGB (3 channels) - legacy support
            #     img = img.convert("RGB")
            #     img = img.resize(self.input_size, Image.BILINEAR)
            #     # Convert to tensor
            #     img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            #     img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            #     # Apply normalization (matching original preprocess: mean=[0.5,0.5,0.5])
            #     mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
            #     std = torch.tensor(self.normalize_std).view(3, 1, 1)
            #     img_tensor = (img_tensor - mean) / std
            
        elif self.model_type == 'htsat':
            # HTSAT handles its own spectrogram processing internally
            # Just return the normalized spectrogram
            img_tensor = torch.from_numpy(S_norm).float()
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return img_tensor
    
    def __call__(self, filepath_or_waveform: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Complete preprocessing pipeline: audio file -> model input tensor.
        
        Args:
            filepath_or_waveform: Either path to audio file or waveform array/tensor
            
        Returns:
            Model input tensor with batch dimension:
            - ResNet50: (1, C, H, W) where C=1 for grayscale, C=3 for RGB
            - HTSAT: (1, mel_bins, time_frames) or as expected by model
        """
        # Load audio if filepath provided
        if isinstance(filepath_or_waveform, str):
            waveform = self.load_audio(filepath_or_waveform)
        elif isinstance(filepath_or_waveform, torch.Tensor):
            waveform = filepath_or_waveform.cpu().numpy()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
        else:
            waveform = np.asarray(filepath_or_waveform)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
        
        # Generate mel-spectrogram
        mel_spec = self.audio_to_spectrogram(waveform)
        
        # Convert to image tensor
        img_tensor = self.spectrogram_to_image(mel_spec)
        
        # Add batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        elif img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        
        return img_tensor

