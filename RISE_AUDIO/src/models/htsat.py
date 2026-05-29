#!/usr/bin/env python
# coding: utf-8

"""HTSAT (Hierarchical Token Semantic Audio Transformer) model wrapper for RISE audio."""

import torch
import torch.nn as nn
import sys
import os
from src.utils import DEFAULT_DATASET
from src.models.base import ModelInputType
from src.preprocessor import RawAudioPreprocessor

# Add HTSAT path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../HTSAT/HTS-Audio-Transformer')))

from model.htsat import HTSAT_Swin_Transformer


class HTSATConfigESC50:
    """Configuration for HTSAT ESC-50 model."""
    
    # Dataset config
    dataset_type = "esc-50"
    classes_num = 50
    
    # Loss config
    loss_type = "clip_ce"
    
    # Model architecture config
    htsat_window_size = 8
    htsat_spec_size = 256
    htsat_patch_size = 4
    htsat_stride = (4, 4)
    htsat_num_head = [4, 8, 16, 32]
    htsat_dim = 96
    htsat_depth = [2, 2, 6, 2]
    
    # Audio processing
    sample_rate = 32000
    clip_samples = 32000 * 10  # 10 seconds
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    
    # Model features
    enable_tscam = True
    
    # Deprecated optimization flags
    htsat_attn_heatmap = False
    htsat_hier_output = False
    htsat_use_max = False
    
    # Data augmentation flags (disabled for inference)
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False


class HTSATConfigUrbanSound8K:
    """Configuration for HTSAT UrbanSound8K model."""
    
    # Dataset config
    dataset_type = "urbansound8k"
    classes_num = 10
    
    # Loss config
    loss_type = "clip_ce"
    
    # Model architecture config (same as ESC-50)
    htsat_window_size = 8
    htsat_spec_size = 256
    htsat_patch_size = 4
    htsat_stride = (4, 4)
    htsat_num_head = [4, 8, 16, 32]
    htsat_dim = 96
    htsat_depth = [2, 2, 6, 2]
    
    # Audio processing (same as ESC-50)
    sample_rate = 32000
    clip_samples = 32000 * 10  # 10 seconds
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    
    # Model features
    enable_tscam = True
    
    # Deprecated optimization flags (REQUIRED by HTSAT model)
    htsat_attn_heatmap = False
    htsat_hier_output = False
    htsat_use_max = False
    
    # Data augmentation flags (disabled for inference)
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False


def get_htsat_config(dataset='esc50'):
    """Get HTSAT config based on dataset."""
    configs = {
        'esc50': HTSATConfigESC50(),
        'urbansound8k': HTSATConfigUrbanSound8K()
    }
    return configs.get(dataset, HTSATConfigESC50())


def HTSATModel(weights_path=None, dataset=None):
    """
    Setup and load the HTSAT model for audio classification.
    
    Args:
        weights_path: Path to checkpoint file (.ckpt or .pth)
        dataset: Dataset type ('esc50' or 'urbansound8k')
        
    Returns:
        Model with predict method and preprocessor
    """
    if dataset is None:
        dataset = DEFAULT_DATASET
    
    if weights_path is None:
        weights_path = 'checkpoints/HTSAT.ckpt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_htsat_config(dataset)
    
    print(f"Loading HTSAT checkpoint for {dataset.upper()}")
    print(f"  Checkpoint path: {weights_path}")
    print(f"  Number of classes: {config.classes_num}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Create HTSAT model with official architecture
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=config.classes_num,
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_head=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Trained model format (.pth) - e.g., UrbanSound8K fine-tuned model
        state_dict = checkpoint['model_state_dict']
        print("  Format: Trained model (model_state_dict)")
        
        # Load weights directly
        missing_keys, unexpected_keys = sed_model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  Warning: {len(missing_keys)} missing keys")
        if len(unexpected_keys) > 0:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")
            
    elif 'state_dict' in checkpoint:
        # AudioSet/ESC-50 format (.ckpt with sed_model prefix)
        state_dict = checkpoint['state_dict']
        print("  Format: Lightning checkpoint (sed_model.* prefix)")
        
        # Extract sed_model weights (remove 'sed_model.' prefix)
        sed_model_state = {}
        for key, value in state_dict.items():
            if key.startswith('sed_model.'):
                new_key = key[10:]
                sed_model_state[new_key] = value
        
        # Load weights
        missing_keys, unexpected_keys = sed_model.load_state_dict(sed_model_state, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  Warning: {len(missing_keys)} missing keys")
        if len(unexpected_keys) > 0:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")
    else:
        # Direct state dict
        print("  Format: Direct state dict")
        sed_model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device
    sed_model = sed_model.to(device)
    
    # Ensure spectrogram extractors are on correct device
    if hasattr(sed_model, 'spectrogram_extractor'):
        sed_model.spectrogram_extractor = sed_model.spectrogram_extractor.to(device)
    if hasattr(sed_model, 'logmel_extractor'):
        sed_model.logmel_extractor = sed_model.logmel_extractor.to(device)
    
    sed_model.eval()
    
    # Freeze parameters
    for p in sed_model.parameters():
        p.requires_grad = False
    
    # Wrap in DataParallel if CUDA available
    if torch.cuda.is_available():
        sed_model = nn.DataParallel(sed_model)
    
    print(f"✓ HTSAT model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"  Checkpoint val accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    # Wrap with predict method
    class ModelWithPredict:
        def __init__(self, model):
            self.model = model
            # Model metadata
            self.input_type = ModelInputType.RAW_AUDIO
            self.sample_rate = config.sample_rate  # 32kHz
            self.input_length = config.clip_samples  # 320,000 samples (10 seconds)
            self.preprocessor = RawAudioPreprocessor(
                target_sample_rate=self.sample_rate,
                fixed_length=self.input_length
            )
        
        def predict(self, input_tensor):
            """
            Predict method that returns logits and probabilities.
            
            Args:
                input_tensor: Raw waveform (batch, samples) at 32kHz
                
            Returns:
                Tuple of (logits, probabilities)
            """
            with torch.no_grad():
                # HTSAT expects (batch, samples) and returns dict with 'clipwise_output'
                # The third argument is infer_mode (True for inference)
                output_dict = self.model(input_tensor, None, True)
                
                # Get clipwise predictions
                if isinstance(output_dict, dict):
                    logits = output_dict['clipwise_output']
                else:
                    logits = output_dict
                
                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)
                
                return logits, probs
        
        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)
        
        def parameters(self):
            return self.model.parameters()
        
        def eval(self):
            self.model.eval()
            return self
    
    return ModelWithPredict(sed_model)


def get_model_predictions(model, input_tensor):
    """Get model predictions for input tensor."""
    with torch.no_grad():
        logits, probs = model.predict(input_tensor)
        predictions, classes = torch.max(probs, dim=1)
    return predictions, classes

