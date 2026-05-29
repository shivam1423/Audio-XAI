"""
Load HTSAT AudioSet checkpoint and adapt for UrbanSound8K
Handles the class mismatch: AudioSet (527 classes) -> UrbanSound8K (10 classes)
"""

import os
import sys
import torch
import torch.nn as nn

# Add official repo to path (should be in parent HTSAT folder)
OFFICIAL_REPO = "../HTSAT/HTS-Audio-Transformer"
if os.path.exists(OFFICIAL_REPO):
    sys.path.insert(0, OFFICIAL_REPO)
else:
    # Try alternative path
    OFFICIAL_REPO = "HTS-Audio-Transformer"
    if os.path.exists(OFFICIAL_REPO):
        sys.path.insert(0, OFFICIAL_REPO)
    else:
        raise FileNotFoundError(
            "HTS-Audio-Transformer repository not found. "
            "Please ensure it exists in ../HTSAT/HTS-Audio-Transformer or HTS-Audio-Transformer/"
        )
# Import official HTSAT model
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper


class HTSATConfig:
    """Configuration matching the official HTSAT AudioSet setup"""
    
    # Dataset config
    dataset_type = "audioset"
    classes_num = 527  # AudioSet has 527 classes
    
    # Loss config
    loss_type = "clip_bce"  # AudioSet uses binary cross-entropy
    
    # Model architecture config (same for all HTSAT models)
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
    
    # Data augmentation flags
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False


def load_audioset_checkpoint_for_urbansound8k(checkpoint_path, device='cuda', num_classes=10):
    """
    Load HTSAT AudioSet checkpoint and adapt for UrbanSound8K
    
    Strategy:
    1. Load AudioSet checkpoint (527 classes)
    2. Extract all weights except final classification head
    3. Replace final head with new one for 10 classes (UrbanSound8K)
    
    Args:
        checkpoint_path: Path to AudioSet checkpoint (.ckpt file)
        device: Device to load on
        num_classes: Number of output classes (10 for UrbanSound8K)
    
    Returns:
        model: Loaded HTSAT model adapted for UrbanSound8K
        config: Configuration object
    """
    config = HTSATConfig()
    
    print(f"Loading AudioSet checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create HTSAT model with AudioSet config (527 classes)
    print("Creating HTSAT model with AudioSet architecture (527 classes)...")
    sed_model_audioset = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=config.classes_num,  # 527 for AudioSet
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_head=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Extract sed_model weights
    sed_model_state = {}
    for key, value in state_dict.items():
        if key.startswith('sed_model.'):
            # Remove 'sed_model.' prefix
            new_key = key[10:]
            sed_model_state[new_key] = value
    
    # Load AudioSet weights (will include 527-class head)
    print("Loading AudioSet weights...")
    missing_keys, unexpected_keys = sed_model_audioset.load_state_dict(
        sed_model_state, strict=False
    )
    
    if len(missing_keys) > 0:
        print(f"Warning: {len(missing_keys)} missing keys (expected for class mismatch)")
        if len(missing_keys) < 20:
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
    
    if len(unexpected_keys) > 0:
        print(f"Warning: {len(unexpected_keys)} unexpected keys")
        if len(unexpected_keys) < 20:
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
    
    # Now create new model for UrbanSound8K (10 classes)
    print(f"Creating new model for UrbanSound8K ({num_classes} classes)...")
    config.classes_num = num_classes
    config.loss_type = "clip_ce"  # UrbanSound8K uses cross-entropy (single label)
    
    sed_model_us8k = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=num_classes,  # 10 for UrbanSound8K
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_head=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )
    
    # Copy all weights except the final classification head
    print("Transferring weights (excluding final classification layer)...")
    us8k_state = sed_model_us8k.state_dict()
    audioset_state = sed_model_audioset.state_dict()
    
    transferred = 0
    skipped = 0
    for key in us8k_state:
        if key in audioset_state:
            # Skip final classification head and tscam_conv (both are class-dependent)
            if 'head' in key or 'classifier' in key or 'tscam_conv' in key:
                skipped += 1
                continue
            # Check if shapes match before transferring
            if us8k_state[key].shape != audioset_state[key].shape:
                print(
                    f"Warning: Shape mismatch for {key}: {us8k_state[key].shape} vs {audioset_state[key].shape}, skipping")
                skipped += 1
                continue
            us8k_state[key] = audioset_state[key]
            transferred += 1
        else:
            print(f"Warning: Key {key} not found in AudioSet checkpoint")

    # Load transferred weights
    sed_model_us8k.load_state_dict(us8k_state, strict=False)
    
    print(f"✓ Transferred {transferred} layers from AudioSet checkpoint")
    print(f"  Skipped {skipped} classification head layers (randomly initialized)")
    print(f"  Final classification layer initialized for {num_classes} classes")
    
    # Move model to device
    sed_model_us8k = sed_model_us8k.to(device)
    
    # Ensure all spectrogram extractor components are on the correct device
    if hasattr(sed_model_us8k, 'spectrogram_extractor'):
        sed_model_us8k.spectrogram_extractor = sed_model_us8k.spectrogram_extractor.to(device)
    if hasattr(sed_model_us8k, 'logmel_extractor'):
        sed_model_us8k.logmel_extractor = sed_model_us8k.logmel_extractor.to(device)
    
    sed_model_us8k.eval()
    
    print(f"✓ Model loaded successfully and adapted for UrbanSound8K")
    if 'epoch' in checkpoint:
        print(f"  Original checkpoint epoch: {checkpoint['epoch']}")
    
    return sed_model_us8k, config

