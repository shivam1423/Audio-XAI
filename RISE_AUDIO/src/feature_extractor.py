"""
DEPRECATED: This module is kept for backward compatibility only.

New code should use model-specific preprocessors from src.preprocessor instead:
- ResNetPreprocessor for ResNet models
- RawAudioPreprocessor for wav2vec2/ACDNet models

These preprocessors are automatically used by the unified saliency generation pipeline.
"""

import warnings
import torch
import torchaudio
import numpy as np
import librosa
from PIL import Image
from typing import List, Tuple
from src.utils import (
    MEL_SR, MEL_N_FFT, MEL_HOP_LENGTH, MEL_N_MELS, DEFAULT_EDGE_SIGMA_PX
)

def audio_tensor_to_mel_spectrogram_image(
        audio_data,
        sr: int = MEL_SR,
        n_fft: int = MEL_N_FFT,
        hop_length: int = MEL_HOP_LENGTH,
        n_mels: int = MEL_N_MELS,
) -> Image.Image:
    """
    DEPRECATED: Use ResNetPreprocessor from src.preprocessor instead.
    
    Convert audio data to mel spectrogram image using librosa (matching testing.py approach).
    
    Args:
        audio_data: Audio tensor or numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel bins
        
    Returns:
        PIL Image (grayscale, 224x224)
    """
    warnings.warn(
        "audio_tensor_to_mel_spectrogram_image is deprecated. "
        "Use ResNetPreprocessor from src.preprocessor instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Convert to numpy array (handle both tensors and arrays)
    if isinstance(audio_data, torch.Tensor):
        if audio_data.dim() > 1:
            audio_np = audio_data.squeeze().detach().cpu().numpy()
        else:
            audio_np = audio_data.detach().cpu().numpy()
    else:
        # It's already a numpy array
        audio_np = np.asarray(audio_data)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
    
    # Create mel spectrogram using librosa (matching testing.py)
    S = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    # Convert to dB scale using librosa (matching testing.py)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to [0, 255] for image conversion (matching testing.py)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_img = (S_norm * 255).astype(np.uint8)
    
    # Create PIL image and resize (matching testing.py)
    img = Image.fromarray(S_img)
    img = img.convert("L")  # grayscale
    img = img.resize((224, 224), Image.BILINEAR)
    
    return img

def process_masked_audio_list(
        masked_audio_list: List[torch.Tensor],
        sr: int = MEL_SR,
        n_fft: int = MEL_N_FFT,
        hop_length: int = MEL_HOP_LENGTH,
        n_mels: int = MEL_N_MELS,
        orig_sr: int = None
) -> List[Image.Image]:
    """
    DEPRECATED: Use model.preprocessor.process_masked_audio_list() instead.
    
    Process list of masked audio tensors to mel spectrogram images using librosa.
    
    This function is kept for backward compatibility. New code should use
    model-specific preprocessors that are automatically applied by the
    unified saliency generation pipeline.
    
    Args:
        masked_audio_list: List of audio tensors
        sr: Target sample rate for mel spectrogram
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel bins
        orig_sr: Original sample rate of audio
        
    Returns:
        List of PIL Images (grayscale, 224x224)
    """
    warnings.warn(
        "process_masked_audio_list is deprecated. "
        "Use model.preprocessor.process_masked_audio_list() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    mel_images = []
    
    for i, audio_tensor in enumerate(masked_audio_list):
        if i % 1000 == 0:
            print(f"Converting audio {i + 1}/{len(masked_audio_list)} to mel spectrogram")

        # Convert to numpy and resample using librosa (matching testing.py)
        y_48k_np = audio_tensor.squeeze().detach().cpu().numpy()
        y_final = librosa.resample(y_48k_np, orig_sr=orig_sr, target_sr=sr)

        # Convert to mel spectrogram image using librosa approach
        mel_img = audio_tensor_to_mel_spectrogram_image(
            y_final, sr, n_fft, hop_length, n_mels
        )
        mel_images.append(mel_img)

    return mel_images