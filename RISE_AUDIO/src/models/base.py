#!/usr/bin/env python
# coding: utf-8

"""Base model interface for unified handling of spectrogram and raw audio models."""

from enum import Enum


class ModelInputType(Enum):
    """Enum for model input types."""
    SPECTROGRAM = "spectrogram"  # Expects mel spectrogram images (ResNet)
    RAW_AUDIO = "raw_audio"      # Expects raw waveforms (wav2vec2, ACDNet, HTSAT)


def get_model_type(model) -> ModelInputType:
    """
    Detect model type from model instance.
    
    This function checks the model's attributes to determine if it expects:
    - SPECTROGRAM: mel spectrogram images (224x224 RGB) for CNN-based models
    - RAW_AUDIO: raw waveforms (1D audio tensors) for transformer-based models
    
    Args:
        model: Model instance (ResNet, wav2vec2, ACDNet, HTSAT, etc.)
        
    Returns:
        ModelInputType enum indicating the input format required
    """
    # Check if model has explicit input_type attribute
    if hasattr(model, 'input_type'):
        return model.input_type
    
    # Fallback: detect from model wrapper class name
    model_class_name = type(model).__name__.lower()
    model_str = str(type(model)).lower()
    
    # Check for raw audio model indicators (wav2vec2, acdnet, htsat)
    if any(name in model_class_name or name in model_str 
           for name in ['wav2vec', 'acdnet', 'htsat']):
        return ModelInputType.RAW_AUDIO
    
    # Check for spectrogram model indicators (ResNet, etc.)
    if any(name in model_class_name or name in model_str 
           for name in ['resnet', 'vgg', 'efficientnet']):
        return ModelInputType.SPECTROGRAM
    
    # Default assumption: spectrogram-based (backward compatibility)
    print("Warning: Could not detect model type, defaulting to SPECTROGRAM")
    return ModelInputType.SPECTROGRAM


def requires_spectrogram(model) -> bool:
    """Check if model requires spectrogram input."""
    return get_model_type(model) == ModelInputType.SPECTROGRAM


def requires_raw_audio(model) -> bool:
    """Check if model requires raw audio input."""
    return get_model_type(model) == ModelInputType.RAW_AUDIO

