#!/usr/bin/env python
# coding: utf-8

"""Model-specific audio preprocessing for RISE framework."""

from abc import ABC, abstractmethod
from typing import List, Union
import torch
import numpy as np
from PIL import Image
import librosa
import torchaudio.transforms as T


class AudioPreprocessor(ABC):
    """Base class for model-specific audio preprocessing."""
    
    @abstractmethod
    def process_masked_audio_list(
        self, 
        masked_audio_list: List[torch.Tensor],
        sample_rate: int
    ) -> List:
        """
        Convert masked audio to model's expected input format.
        
        Args:
            masked_audio_list: List of masked audio tensors
            sample_rate: Sample rate of the masked audio
            
        Returns:
            List of preprocessed inputs in model-specific format
        """
        pass
    
    @abstractmethod
    def process_original_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ):
        """
        Process original (unmasked) audio for target class prediction.
        
        Args:
            audio: Original audio tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Preprocessed input in model-specific format
        """
        pass
    
    @property
    @abstractmethod
    def target_sample_rate(self) -> int:
        """Target sample rate for this model."""
        pass


class ResNetPreprocessor(AudioPreprocessor):
    """
    ResNet expects 224x224 RGB mel spectrogram images with ImageNet normalization.
    
    Pipeline:
    1. Resample audio to 22050 Hz
    2. Generate mel spectrogram (128 mel bins)
    3. Convert to dB scale
    4. Normalize to [0, 255]
    5. Create PIL Image and resize to 224x224
    6. Convert to RGB (3 channels)
    7. Apply ImageNet normalization
    """
    
    def __init__(self):
        self._target_sr = 22050
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 512
        self.image_size = (224, 224)
    
    @property
    def target_sample_rate(self) -> int:
        return self._target_sr
    
    def process_masked_audio_list(
        self, 
        masked_audio_list: List[torch.Tensor],
        sample_rate: int
    ) -> List[Image.Image]:
        """
        Convert masked audio to mel spectrogram PIL Images.
        
        Returns:
            List of PIL Images (grayscale, 224x224)
        """
        mel_images = []
        
        # Create resampler once
        if sample_rate != self._target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self._target_sr)
        else:
            resampler = None
        
        for i, audio_tensor in enumerate(masked_audio_list):
            if i % 1000 == 0:
                print(f"Converting audio {i + 1}/{len(masked_audio_list)} to mel spectrogram")
            
            # Resample if needed
            if resampler is not None:
                y_np = audio_tensor.squeeze().detach().cpu().numpy()
                y_resampled = librosa.resample(y_np, orig_sr=sample_rate, target_sr=self._target_sr)
            else:
                y_resampled = audio_tensor.squeeze().detach().cpu().numpy()
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y_resampled,
                sr=self._target_sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            
            # Convert to dB scale
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 255]
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            mel_img_array = (mel_norm * 255).astype(np.uint8)
            
            # Create PIL Image and resize
            img = Image.fromarray(mel_img_array)
            img = img.convert("L")  # Ensure grayscale
            img = img.resize(self.image_size, Image.BILINEAR)
            
            mel_images.append(img)
        
        return mel_images
    
    def process_original_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> Image.Image:
        """Process single audio to mel spectrogram image."""
        return self.process_masked_audio_list([audio], sample_rate)[0]


class RawAudioPreprocessor(AudioPreprocessor):
    """
    Preprocessor for models that take raw waveforms (wav2vec2, ACDNet).
    
    Pipeline:
    1. Resample audio to model's target sample rate
    2. Pad or truncate to fixed length (if required)
    3. Return raw waveform tensors
    """
    
    def __init__(self, target_sample_rate: int = 16000, fixed_length: int = None):
        """
        Initialize raw audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate (default: 16000 for wav2vec2)
            fixed_length: Fixed length in samples (None = variable length)
        """
        self._target_sr = target_sample_rate
        self.fixed_length = fixed_length
    
    @property
    def target_sample_rate(self) -> int:
        return self._target_sr
    
    def _pad_or_truncate(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or truncate audio to fixed length."""
        if self.fixed_length is None:
            return audio
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        current_length = audio.shape[1]
        
        if current_length > self.fixed_length:
            # Center crop
            start = (current_length - self.fixed_length) // 2
            audio = audio[:, start:start + self.fixed_length]
        elif current_length < self.fixed_length:
            # Pad with zeros
            pad_amount = self.fixed_length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        
        return audio
    
    def process_masked_audio_list(
        self, 
        masked_audio_list: List[torch.Tensor],
        sample_rate: int
    ) -> List[torch.Tensor]:
        """
        Convert masked audio to raw waveforms at target sample rate.
        
        Returns:
            List of waveform tensors
        """
        # Create resampler once if needed
        if sample_rate != self._target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self._target_sr)
        else:
            resampler = None
        
        processed_waveforms = []
        
        for i, audio_tensor in enumerate(masked_audio_list):
            if i % 1000 == 0:
                print(f"Processing audio {i + 1}/{len(masked_audio_list)}")
            
            # Resample if needed
            if resampler is not None:
                audio_resampled = resampler(audio_tensor)
            else:
                audio_resampled = audio_tensor
            
            # Pad or truncate to fixed length
            audio_processed = self._pad_or_truncate(audio_resampled)
            
            processed_waveforms.append(audio_processed)
        
        return processed_waveforms
    
    def process_original_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """Process single audio waveform."""
        return self.process_masked_audio_list([audio], sample_rate)[0]

