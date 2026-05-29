"""
Audio preprocessing utilities for UrbanSound8K
Handles resampling, normalization, and data preparation for ACDNet
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocessor for audio data - handles resampling and normalization"""
    
    def __init__(self, target_sr=20000, target_length=30000):
        """
        Initialize preprocessor
        Args:
            target_sr: Target sampling rate (20kHz for ACDNet)
            target_length: Target audio length in samples (1.5s at 20kHz = 30000)
        """
        self.target_sr = target_sr
        self.target_length = target_length
    
    def load_audio(self, audio_path, sr=None):
        """
        Load audio file
        Args:
            audio_path: Path to audio file
            sr: Target sampling rate (if None, uses target_sr)
        Returns:
            Audio waveform as numpy array
        """
        if sr is None:
            sr = self.target_sr
        
        try:
            # Load audio file with librosa
            waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
            return waveform
        except Exception as e:
            raise Exception(f"Error loading audio file {audio_path}: {e}")
    
    def resample(self, waveform, orig_sr, target_sr=None):
        """
        Resample audio to target sampling rate
        Args:
            waveform: Input audio waveform
            orig_sr: Original sampling rate
            target_sr: Target sampling rate (if None, uses self.target_sr)
        Returns:
            Resampled waveform
        """
        if target_sr is None:
            target_sr = self.target_sr
        
        if orig_sr == target_sr:
            return waveform
        
        # Resample using librosa
        resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        return resampled
    
    def pad_or_truncate(self, waveform, target_length=None):
        """
        Pad or truncate audio to target length
        Args:
            waveform: Input audio waveform
            target_length: Target length in samples (if None, uses self.target_length)
        Returns:
            Padded or truncated waveform
        """
        if target_length is None:
            target_length = self.target_length
        
        current_length = len(waveform)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = np.pad(waveform, (0, padding), mode='constant', constant_values=0)
        elif current_length > target_length:
            # Truncate (take first target_length samples)
            waveform = waveform[:target_length]
        
        return waveform
    
    def normalize(self, waveform, method='peak'):
        """
        Normalize audio waveform
        Args:
            waveform: Input audio waveform
            method: Normalization method ('peak' or 'standard')
        Returns:
            Normalized waveform
        """
        if method == 'peak':
            # Normalize to [-1, 1] range based on peak
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val
        elif method == 'standard':
            # Standardize to zero mean and unit variance
            mean = np.mean(waveform)
            std = np.std(waveform)
            if std > 0:
                waveform = (waveform - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return waveform
    
    def preprocess(self, audio_path, normalize_method='peak'):
        """
        Complete preprocessing pipeline
        Args:
            audio_path: Path to audio file
            normalize_method: Normalization method
        Returns:
            Preprocessed waveform ready for model input
        """
        # Load audio at target sampling rate
        waveform = self.load_audio(audio_path, sr=self.target_sr)
        
        # Pad or truncate to target length
        waveform = self.pad_or_truncate(waveform)
        
        # Normalize
        waveform = self.normalize(waveform, method=normalize_method)
        
        return waveform
    
    def preprocess_for_training(self, audio_path, augment_funcs=None):
        """
        Preprocessing pipeline for training (with optional augmentation)
        Args:
            audio_path: Path to audio file
            augment_funcs: List of augmentation functions to apply
        Returns:
            Preprocessed waveform
        """
        # Load audio
        waveform = self.load_audio(audio_path, sr=self.target_sr)
        
        # Apply augmentation functions if provided
        if augment_funcs is not None:
            for func in augment_funcs:
                waveform = func(waveform)
        
        # Pad or truncate
        waveform = self.pad_or_truncate(waveform)
        
        # Normalize
        waveform = self.normalize(waveform, method='peak')
        
        return waveform
    
    def prepare_for_model(self, waveform):
        """
        Prepare waveform for ACDNet model input
        Args:
            waveform: Preprocessed waveform (1D array)
        Returns:
            Waveform reshaped for model input (1, 1, length, 1)
        """
        # ACDNet expects shape: (batch, channels, height, width)
        # For raw waveform: (batch, 1, 1, length)
        waveform = np.expand_dims(waveform, axis=0)  # Add channel dimension
        waveform = np.expand_dims(waveform, axis=0)  # Add height dimension
        waveform = np.expand_dims(waveform, axis=-1)  # Add width dimension
        return waveform


def create_test_crops(waveform, input_length, n_crops):
    """
    Create multiple crops from audio for testing (10-crop evaluation)
    Args:
        waveform: Input audio waveform
        input_length: Length of each crop
        n_crops: Number of crops to create
    Returns:
        Array of cropped waveforms
    """
    if len(waveform) < input_length:
        # Pad if too short
        waveform = np.pad(waveform, (0, input_length - len(waveform)), mode='constant')
    
    # Calculate stride for evenly spaced crops
    if len(waveform) == input_length:
        # If exact length, just return single crop repeated
        crops = np.array([waveform for _ in range(n_crops)])
    else:
        stride = (len(waveform) - input_length) // (n_crops - 1)
        crops = []
        for i in range(n_crops):
            start = stride * i
            end = start + input_length
            crops.append(waveform[start:end])
        crops = np.array(crops)
    
    return crops


def convert_to_mono(waveform):
    """
    Convert stereo audio to mono
    Args:
        waveform: Audio waveform (can be 1D or 2D)
    Returns:
        Mono waveform (1D)
    """
    if waveform.ndim == 2:
        # Average across channels
        waveform = np.mean(waveform, axis=0)
    return waveform
