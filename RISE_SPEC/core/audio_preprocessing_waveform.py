#!/usr/bin/env python
# coding: utf-8

"""Audio preprocessing for waveform-based models (Wav2Vec2)."""

import torch
try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None
import librosa
import numpy as np
from typing import Union


class WaveformPreprocessor:
    """
    Preprocessor for waveform-based models (e.g., Wav2Vec2).
    Handles loading, resampling, and padding/truncating audio to target length.
    """
    
    def __init__(
        self,
        target_sr: int,
        target_length: int
    ):
        """
        Initialize waveform preprocessor.
        
        Args:
            target_sr: Target sample rate (e.g., 16000 for Wav2Vec2)
            target_length: Target length in samples (e.g., 80000 for 5s @ 16kHz)
        """
        self.target_sr = target_sr
        self.target_length = target_length
        
    def load_audio(self, filepath: str) -> torch.Tensor:
        """
        Load audio file and convert to mono at target sample rate.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio waveform as torch tensor (1D)
        """
        try:
            if torchaudio is None:
                raise RuntimeError("torchaudio not available")
            waveform, sr = torchaudio.load(filepath)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
            
            # Convert to 1D
            waveform = waveform.squeeze(0)
            
        except Exception as e:
            print(f"torchaudio failed ({e}), trying librosa...")
            # Fallback to librosa
            waveform_np, _ = librosa.load(filepath, sr=self.target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np).float()
        
        return waveform
    
    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample waveform to target sample rate.
        
        Args:
            waveform: Input waveform
            orig_sr: Original sample rate
            
        Returns:
            Resampled waveform
        """
        if orig_sr == self.target_sr:
            return waveform
        
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for resampling when orig_sr != target_sr")
        resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return resampler(waveform)
    
    def pad_or_truncate(self, waveform: torch.Tensor, target_length: int = None) -> torch.Tensor:
        """
        Pad or truncate waveform to target length.
        
        Args:
            waveform: Input waveform
            target_length: Target length (uses self.target_length if None)
            
        Returns:
            Waveform with target length
        """
        if target_length is None:
            target_length = self.target_length
        
        current_length = waveform.shape[0] if waveform.dim() == 1 else waveform.shape[1]
        
        if current_length < target_length:
            # Pad by repeating the audio
            num_repeats = int(np.ceil(target_length / current_length))
            waveform = waveform.repeat(num_repeats) if waveform.dim() == 1 else waveform.repeat(1, num_repeats)
        
        # Truncate to exact length
        if waveform.dim() == 1:
            waveform = waveform[:target_length]
        else:
            waveform = waveform[:, :target_length]
        
        return waveform
    
    def __call__(self, filepath_or_waveform: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Complete preprocessing pipeline: audio file -> model input tensor.
        
        Args:
            filepath_or_waveform: Either path to audio file or waveform array/tensor
            
        Returns:
            Model input tensor with batch dimension: (1, target_length)
        """
        # Load audio if filepath provided
        if isinstance(filepath_or_waveform, str):
            waveform = self.load_audio(filepath_or_waveform)
        elif isinstance(filepath_or_waveform, np.ndarray):
            waveform = torch.from_numpy(filepath_or_waveform).float()
        else:
            waveform = filepath_or_waveform
        
        # Ensure 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        # Pad or truncate to target length
        waveform = self.pad_or_truncate(waveform)
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0)  # (target_length,) -> (1, target_length)
        
        return waveform

