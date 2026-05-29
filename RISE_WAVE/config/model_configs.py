#!/usr/bin/env python
# coding: utf-8

"""Model-specific configurations for audio preprocessing and model loading."""

import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configurations
MODEL_CONFIGS = {
    'resnet50': {
        'type': 'spectrogram',
        'weights_path': 'checkpoints/resnet50_urbansound8k.pth',
        'sample_rate': 22050,
        'n_fft': 1024,          # Matches Resnet50_UrbanSound8K training
        'hop_length': 512,
        'n_mels': 128,
        'fmin': 0,
        'fmax': 11025,          # sr/2 = 11025 Hz (fmax=None in training → Nyquist)
        'input_size': (224, 224),  # Matches Resnet50_UrbanSound8K training
        'num_classes': 10,
        'normalize_mean': [0.5, 0.5, 0.5],  # 3-channel input (grayscale replicated)
        'normalize_std': [0.5, 0.5, 0.5],
        'clip_samples': 88200,  # 4 seconds at 22050 Hz
        'duration': 4.0  # Audio duration in seconds
    },
    'htsat': {
        'type': 'waveform',
        # 'weights_path': os.path.join(BASE_DIR, '..', 'HTSAT', 'HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt'),# for esc50
        'weights_path': 'checkpoints/htsat_us8k.pth',
        'sample_rate': 32000,
        'target_length': 320000,
        'n_fft': 1024,
        'hop_length': 320,
        'n_mels': 64,
        'fmin': 50,
        'fmax': 14000,
        'clip_samples': 320000,  # 10 seconds at 32kHz
        'input_size': (64, 1001),  # mel_bins x time_frames (approximate)
        # 'num_classes': 50,# for esc50
        'num_classes': 10,
        'window_size': 1024,
        'patch_size': 4,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size_spec': 8
    },
    'wav2vec2': {
        'type': 'waveform',
        'weights_path': 'checkpoints/best_model_wav2vec2_us8k.pt',
        'sample_rate': 16000,
        'target_length': 64000,  # 4 seconds at 16kHz for UrbanSound8K (default)
        'num_classes': 10,
        'model_name': 'facebook/wav2vec2-base'
    },
    'acdnet': {
        'type': 'waveform',
        'weights_path': 'checkpoints/acdnet_us8k_best.pt',
        'sample_rate': 20000,
        'target_length': 30225,  # ~1.5 seconds at 20kHz
        'num_classes': 10
    }
}


def get_model_config(model_type):
    """
    Get configuration for a specific model type.
    
    Args:
        model_type: str, one of 'resnet50', 'htsat', 'wav2vec2'
        
    Returns:
        dict: Model configuration
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_type].copy()


def is_spectrogram_model(model_type):
    """
    Check if model requires spectrogram input.
    
    Args:
        model_type: str
        
    Returns:
        bool: True if model uses spectrograms, False if waveform
    """
    config = get_model_config(model_type)
    return config['type'] == 'spectrogram'


def is_waveform_model(model_type):
    """
    Check if model requires waveform input.
    
    Args:
        model_type: str
        
    Returns:
        bool: True if model uses waveforms, False if spectrogram
    """
    config = get_model_config(model_type)
    return config['type'] == 'waveform'


def get_supported_models():
    """
    Get list of supported model types.
    
    Returns:
        list: List of supported model type strings
    """
    return list(MODEL_CONFIGS.keys())


# Dataset-specific weights, class counts, and target lengths per model.
# These override the defaults in MODEL_CONFIGS when a dataset is specified.
MODEL_DATASET_OVERRIDES = {
    'wav2vec2': {
        'esc50': {
            'num_classes': 50,
            'target_length': 80000,  # 5 seconds at 16kHz for ESC50
            'weights_path': 'checkpoints/best_model_wav2vec2_esc50.pt',
        },
        'urbansound8k': {
            'num_classes': 10,
            'target_length': 64000,  # 4 seconds at 16kHz for UrbanSound8K
            'weights_path': 'checkpoints/best_model_wav2vec2_us8k.pt',
        },
    },
    'acdnet': {
        'esc50': {
            'num_classes': 50,
            'weights_path': 'checkpoints/acdnet_esc50.pt',
        },
        'urbansound8k': {
            'num_classes': 10,
            'weights_path': 'checkpoints/acdnet_us8k_best.pt',
        },
    },
    'resnet50': {
        'esc50':        {'num_classes': 50, 'weights_path': 'checkpoints/resnet50_esc50.pth'},
        'urbansound8k': {'num_classes': 10, 'weights_path': 'checkpoints/resnet50_urbansound8k.pth'},
    },
    'htsat': {
        'esc50':        {'num_classes': 50, 'weights_path': 'checkpoints/htsat_esc50.ckpt'},
        'urbansound8k': {'num_classes': 10, 'weights_path': 'checkpoints/htsat_us8k.pth'},
    },
}


def get_model_config_for_dataset(model_type: str, dataset_name: str) -> dict:
    """Get model config merged with dataset-specific num_classes, target_length, and weights_path.

    Args:
        model_type: 'wav2vec2', 'acdnet', 'resnet50', or 'htsat'
        dataset_name: 'esc50' or 'urbansound8k'

    Returns:
        dict: Model config with dataset-appropriate values applied.

    Raises:
        ValueError: If model_type or dataset_name is not recognized.
    """
    config = get_model_config(model_type)
    overrides = MODEL_DATASET_OVERRIDES.get(model_type, {}).get(dataset_name, {})
    if not overrides:
        raise ValueError(
            f"No dataset overrides found for model='{model_type}', dataset='{dataset_name}'. "
            f"Supported datasets: {list(MODEL_DATASET_OVERRIDES.get(model_type, {}).keys())}"
        )
    config.update(overrides)
    return config


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration.

    Args:
        dataset_name: 'esc50' or 'urbansound8k'

    Returns:
        dict with keys: num_classes, default_audio_dir, default_spectrogram_dir

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name == 'esc50':
        return {
            'num_classes': 50,
            'default_audio_dir': '../ESC50/audio',
            'default_spectrogram_dir': 'ESC50_spectrograms',
        }
    elif dataset_name == 'urbansound8k':
        return {
            'num_classes': 10,
            'default_audio_dir': '../UrbanSound8K/audio/fold10',
            'default_spectrogram_dir': 'UrbanSound8K_spectrograms',
        }
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported datasets: ['esc50', 'urbansound8k']"
        )

