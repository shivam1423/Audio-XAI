#!/usr/bin/env python
# coding: utf-8

"""Model factory for loading and creating different model types."""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.model_configs import get_model_config, is_spectrogram_model
from core.audio_preprocessing_spectrogram import SpectrogramPreprocessor
from core.audio_preprocessing_waveform import WaveformPreprocessor
from core.htsat_spectrogram import (
    HTSATSpectrogramPreprocessor,
    forward_htsat_from_spectrogram,
)


class HTSATWrapper(nn.Module):
    """
    Wrapper to extract clipwise_output from HTSAT's dictionary output.

    HTSAT models return a dict with multiple outputs:
    - 'clipwise_output': Clip-level predictions (used for classification)
    - 'framewise_output': Frame-level predictions (for sound event detection)
    - 'embedding': Feature embeddings

    This wrapper extracts just the clipwise_output for downstream processing.
    """

    def __init__(self, htsat_model):
        super().__init__()
        self.htsat = htsat_model

    def forward(self, x):
        output = self.htsat(x)
        if isinstance(output, dict):
            # Extract clipwise predictions for classification
            return output['clipwise_output']
        # Fallback if model doesn't return dict (shouldn't happen with official HTSAT)
        return output


class HTSATSpectrogramWrapper(nn.Module):
    """Wrapper that accepts mel-spectrograms and bypasses HTSAT's waveform front-end."""

    def __init__(self, htsat_model):
        super().__init__()
        self.htsat = htsat_model

    def forward(self, mel_tensor):
        if mel_tensor.dim() != 4:
            raise ValueError("HTSAT spectrogram wrapper expects 4D tensors (B, C, H, W).")
        # Convert (B, 1, mel_bins, time_frames) -> (B, 1, time_frames, mel_bins)
        mel_for_model = mel_tensor.permute(0, 1, 3, 2).contiguous()
        output = forward_htsat_from_spectrogram(self.htsat, mel_for_model, infer_mode=True)
        if isinstance(output, dict):
            return output['clipwise_output']
        return output


def create_model(
    model_type: str,
    weights_path: str = None,
    num_classes: int = 50,
    htsat_spectrogram_mode: bool = False,
):
    """
    Create and load a model based on model type.
    
    Args:
        model_type: Type of model ('resnet50', 'htsat', 'wav2vec2')
        weights_path: Path to model weights (None = use default from config)
        num_classes: Number of output classes
        
    Returns:
        model: Loaded model wrapped in nn.Sequential with Softmax and DataParallel
    """
    config = get_model_config(model_type)
    
    # Use config weights path if not provided
    if weights_path is None:
        weights_path = config['weights_path']
    
    # Create backbone based on model type
    if model_type == 'resnet50':
        backbone = _create_resnet50(weights_path, num_classes)
    elif model_type == 'htsat':
        backbone = _create_htsat(weights_path, num_classes, config)
        if htsat_spectrogram_mode:
            backbone = HTSATSpectrogramWrapper(backbone)
        else:
            backbone = HTSATWrapper(backbone)
    elif model_type == 'wav2vec2':
        backbone = _create_wav2vec2(weights_path, num_classes, config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create model with softmax
    model = nn.Sequential(backbone, nn.Softmax(dim=1)).cuda()
    model.eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    # Wrap in DataParallel
    model = nn.DataParallel(model)
    
    return model


def _create_resnet50(weights_path: str, num_classes: int):
    """Create ResNet50 backbone for spectrograms."""
    # region agent log
    import json; import time
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cursor', 'debug.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    abs_weights_path = os.path.abspath(weights_path)
    file_exists = os.path.exists(abs_weights_path)
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'initial','hypothesisId':'D','location':'model_factory.py:118','message':'Function entry','data':{'weights_path':weights_path,'abs_path':abs_weights_path,'file_exists':file_exists,'num_classes':num_classes},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # region agent log
    # Add Resnet50_UrbanSound8K to sys.path to resolve config.config imports
    import sys
    resnet50_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'Resnet50_UrbanSound8K')
    resnet50_dir = os.path.abspath(resnet50_dir)
    original_path = sys.path.copy()
    if resnet50_dir not in sys.path:
        sys.path.insert(0, resnet50_dir)
    
    # Check for config package structure
    config_dir = os.path.join(resnet50_dir, 'config')
    config_init = os.path.join(config_dir, '__init__.py')
    config_py = os.path.join(config_dir, 'config.py')
    has_config_init = os.path.exists(config_init)
    has_config_py = os.path.exists(config_py)
    
    # Check if 'config' is already in sys.modules (potential collision)
    config_in_modules = 'config' in sys.modules
    config_module_path = None
    if config_in_modules:
        try:
            config_module_path = sys.modules['config'].__file__
        except:
            config_module_path = 'no __file__ attribute'
    
    # Try to import config.config explicitly to test if it's accessible
    can_import = False
    import_error = None
    try:
        # Force reload if config is already loaded
        if 'config' in sys.modules:
            del sys.modules['config']
        if 'config.config' in sys.modules:
            del sys.modules['config.config']
        import config.config
        can_import = True
    except Exception as e:
        import_error = str(e)
    
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'F,G,I,K','location':'model_factory.py:132','message':'Added Resnet50_UrbanSound8K to sys.path and tested import','data':{'resnet50_dir':resnet50_dir,'dir_exists':os.path.exists(resnet50_dir),'has_config_init':has_config_init,'has_config_py':has_config_py,'config_was_in_modules':config_in_modules,'config_module_path':config_module_path,'can_import':can_import,'import_error':import_error},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # region agent log
    # Try loading with weights_only=False first to inspect structure
    try:
        checkpoint = torch.load(abs_weights_path, map_location='cuda', weights_only=False)
        checkpoint_type = str(type(checkpoint).__name__)
        is_dict = isinstance(checkpoint, dict)
        keys = list(checkpoint.keys()) if is_dict else []
        first_5_keys = keys[:5] if is_dict else []
        has_model_state_dict = 'model_state_dict' in keys if is_dict else False
        has_config = 'config' in keys if is_dict else False
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'initial','hypothesisId':'A,B,E','location':'model_factory.py:124','message':'Checkpoint loaded with weights_only=False','data':{'checkpoint_type':checkpoint_type,'is_dict':is_dict,'num_keys':len(keys) if is_dict else 0,'first_5_keys':first_5_keys,'has_model_state_dict':has_model_state_dict,'has_config':has_config},'timestamp':int(time.time()*1000)})+'\n')
    except Exception as e:
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'initial','hypothesisId':'E','location':'model_factory.py:124','message':'Failed to load with weights_only=False','data':{'error':str(e)},'timestamp':int(time.time()*1000)})+'\n')
        raise
    # endregion
    
    # region agent log
    # Extract state dict based on structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'initial','hypothesisId':'A','location':'model_factory.py:135','message':'Extracted model_state_dict from checkpoint','data':{'num_state_keys':len(state.keys())},'timestamp':int(time.time()*1000)})+'\n')
    else:
        state = checkpoint
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'initial','hypothesisId':'A','location':'model_factory.py:135','message':'Using checkpoint directly as state dict','data':{'is_dict':isinstance(state, dict)},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # region agent log
    # Check for module. and resnet. prefixes
    first_key = next(iter(state)) if isinstance(state, dict) else None
    has_module_prefix = first_key.startswith('module.') if first_key else False
    has_resnet_prefix = first_key.startswith('resnet.') if first_key else False
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'C,L','location':'model_factory.py:125','message':'Checking for prefixes','data':{'first_key':first_key,'has_module_prefix':has_module_prefix,'has_resnet_prefix':has_resnet_prefix},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # Strip module. prefix if present (from DataParallel)
    if isinstance(state, dict) and next(iter(state)).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
        # region agent log
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'C','location':'model_factory.py:145','message':'Stripped module prefix','data':{'new_first_key':next(iter(state))},'timestamp':int(time.time()*1000)})+'\n')
        # endregion
    
    # Strip resnet. prefix if present (from model wrapper)
    if isinstance(state, dict) and next(iter(state)).startswith('resnet.'):
        state = {k.replace('resnet.', ''): v for k, v in state.items()}
        # region agent log
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'L','location':'model_factory.py:208','message':'Stripped resnet prefix','data':{'new_first_key':next(iter(state))},'timestamp':int(time.time()*1000)})+'\n')
        # endregion
    
    # region agent log
    # Detect input channels from checkpoint conv1.weight
    input_channels = 3  # default
    if 'conv1.weight' in state:
        input_channels = state['conv1.weight'].shape[1]
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'M','location':'model_factory.py:215','message':'Detected input channels from checkpoint','data':{'input_channels':input_channels},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # Create ResNet50 with appropriate input channels
    backbone = models.resnet50(weights=None)
    if input_channels != 3:
        # Modify first conv layer for grayscale input
        backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    
    # region agent log
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'M','location':'model_factory.py:225','message':'Created backbone with modified input channels','data':{'input_channels':input_channels,'conv1_in_channels':backbone.conv1.in_channels},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    # region agent log
    try:
        backbone.load_state_dict(state)
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'A,B,C','location':'model_factory.py:150','message':'State dict loaded successfully','data':{'success':True},'timestamp':int(time.time()*1000)})+'\n')
    except Exception as e:
        with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'A,B,C','location':'model_factory.py:150','message':'Failed to load state dict','data':{'error':str(e),'error_type':str(type(e).__name__)},'timestamp':int(time.time()*1000)})+'\n')
        raise
    # endregion
    
    # region agent log
    # Restore original sys.path
    sys.path = original_path
    with open(log_file, 'a') as f: f.write(json.dumps({'sessionId':'debug-session','runId':'post-fix','hypothesisId':'E','location':'model_factory.py:190','message':'Restored sys.path and returning backbone','data':{'success':True},'timestamp':int(time.time()*1000)})+'\n')
    # endregion
    
    print(f"ResNet50 loaded from {weights_path}")
    return backbone


def _create_htsat(weights_path: str, num_classes: int, config: dict):
    """Create HTSAT backbone using official architecture."""

    import sys
    htsat_repo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'HTSAT', 'HTS-Audio-Transformer')
    htsat_repo_path = os.path.abspath(htsat_repo_path)

    # Temporarily prioritize official HTSAT path
    original_path = sys.path.copy()
    sys.path.insert(0, htsat_repo_path)

    try:
        from model.htsat import HTSAT_Swin_Transformer

        # Create HTSATConfig class matching the checkpoint
        class HTSATConfig:
            """Configuration matching the official HTSAT ESC-50 setup"""
            dataset_type = "esc-50"
            classes_num = num_classes
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
            clip_samples = 320000  # 10 seconds
            window_size = 1024
            hop_size = 320
            mel_bins = 64
            fmin = 50
            fmax = 14000
            
            # Model features
            enable_tscam = True
            htsat_attn_heatmap = False
            htsat_hier_output = False
            htsat_use_max = False
            
            # Data augmentation flags
            enable_token_label = False
            enable_time_shift = False
            enable_label_enhance = False
            enable_repeat_mode = False
        
        htsat_config = HTSATConfig()
        
        # Create HTSAT model with exact same config as training
        sed_model = HTSAT_Swin_Transformer(
            spec_size=htsat_config.htsat_spec_size,
            patch_size=htsat_config.htsat_patch_size,
            patch_stride=htsat_config.htsat_stride,
            num_classes=htsat_config.classes_num,
            embed_dim=htsat_config.htsat_dim,
            depths=htsat_config.htsat_depth,
            num_head=htsat_config.htsat_num_head,
            window_size=htsat_config.htsat_window_size,
            config=htsat_config,
        )
        
        # Load checkpoint
        if os.path.exists(weights_path):
            print(f"Loading HTSAT checkpoint from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cuda')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Trained model format (.pth) - e.g., UrbanSound8K fine-tuned model
                state_dict = checkpoint['model_state_dict']
                print("  Format: Trained model (model_state_dict)")
                
                # Load weights directly (no prefix stripping needed)
                missing_keys, unexpected_keys = sed_model.load_state_dict(state_dict, strict=False)
                
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
            else:
                # Direct state dict
                print("  Format: Direct state dict")
                missing_keys, unexpected_keys = sed_model.load_state_dict(checkpoint, strict=False)
            
            if len(missing_keys) > 0:
                print(f"Warning: {len(missing_keys)} missing keys in HTSAT")
                if len(missing_keys) < 20:
                    for key in missing_keys:
                        print(f"  - {key}")
            
            if len(unexpected_keys) > 0:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in HTSAT")
                if len(unexpected_keys) < 20:
                    for key in unexpected_keys:
                        print(f"  - {key}")
            
            print(f"✓ HTSAT loaded from {weights_path}")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
        else:
            print(f"Warning: HTSAT weights not found at {weights_path}, using random initialization")
        sys.path = original_path
        return sed_model
        
    except ImportError as e:
        raise ImportError(
            f"Failed to import official HTSAT: {e}\n"
            f"Make sure HTS-Audio-Transformer repository is at ../HTSAT/HTS-Audio-Transformer/\n"
            f"Clone it with: git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git"
        )


def _create_wav2vec2(weights_path: str, num_classes: int, config: dict):
    """Create Wav2Vec2 backbone."""
    try:
        # Import Wav2Vec2 from parent directory
        wav2vec2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Wav2Vec2')
        sys.path.insert(0, wav2vec2_path)
        
        from model.wav2vec2_classifier import Wav2Vec2Classifier
        
        # Create Wav2Vec2 classifier
        backbone = Wav2Vec2Classifier(
            model_name=config.get('model_name', 'facebook/wav2vec2-base'),
            num_classes=num_classes
        )
        
        # Load checkpoint if exists
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cuda')
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove module prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Wav2Vec2 loaded from {weights_path}")
        else:
            print(f"Warning: Wav2Vec2 weights not found at {weights_path}, using pretrained base model")
        
        return backbone
        
    except ImportError as e:
        raise ImportError(f"Failed to import Wav2Vec2: {e}. Make sure Wav2Vec2 directory is in parent folder.")


def get_target_layer_for_gradcam(model, model_type: str):
    """
    Get the target layer for GradCAM based on model type.
    
    Args:
        model: The model instance (can be wrapped in DataParallel)
        model_type: Type of model ('resnet50', 'htsat', 'wav2vec2')
        
    Returns:
        list: List of target layers for GradCAM
    """
    # Unwrap DataParallel if needed
    if hasattr(model, 'module'):
        unwrapped = model.module
    else:
        unwrapped = model
    
    # Unwrap Sequential if needed
    if isinstance(unwrapped, nn.Sequential):
        backbone = unwrapped[0]
    else:
        backbone = unwrapped
    
    if model_type == 'resnet50':
        # For ResNet50, use the last layer of layer4
        return [backbone.layer4[-1]]
    
    elif model_type == 'htsat':
        # For official HTSAT_Swin_Transformer, use the last swin layer
        if hasattr(backbone, 'layers'):
            # layers is a ModuleList of Swin transformer blocks
            return [backbone.layers[-1]]
        elif hasattr(backbone, 'stages'):
            # Alternative: use stages if available
            return [backbone.stages[-1]]
        else:
            # Fallback to norm layer
            if hasattr(backbone, 'norm'):
                return [backbone.norm]
            # Last resort: use the whole backbone
            return [backbone]
    
    elif model_type == 'wav2vec2':
        # For Wav2Vec2, use the feature encoder
        if hasattr(backbone, 'wav2vec2'):
            if hasattr(backbone.wav2vec2, 'encoder'):
                return [backbone.wav2vec2.encoder.layers[-1]]
            return [backbone.wav2vec2.feature_extractor]
        return [backbone]
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_preprocessor(
    model_type: str,
    config: dict = None,
    htsat_spectrogram_mode: bool = False,
):
    """
    Get the appropriate preprocessor for the model type.
    
    Args:
        model_type: Type of model ('resnet50', 'htsat', 'wav2vec2')
        config: Model configuration (None = load from model_configs)
        
    Returns:
        Preprocessor instance (SpectrogramPreprocessor or WaveformPreprocessor)
    """
    if config is None:
        config = get_model_config(model_type)
    
    if model_type == 'htsat' and htsat_spectrogram_mode:
        preprocessor = HTSATSpectrogramPreprocessor(
            sample_rate=config['sample_rate'],
            window_size=config['n_fft'],
            hop_size=config['hop_length'],
            mel_bins=config['n_mels'],
            fmin=config.get('fmin', 50),
            fmax=config.get('fmax', 14000),
            clip_samples=config.get('clip_samples', config.get('target_length', 320000)),
        )
    elif is_spectrogram_model(model_type):
        # Create spectrogram preprocessor
        preprocessor = SpectrogramPreprocessor(
            model_type=model_type,
            sample_rate=config['sample_rate'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_mels=config['n_mels'],
            fmin=config.get('fmin', 0),
            fmax=config.get('fmax', None),
            input_size=config.get('input_size', (224, 224)),
            normalize_mean=config.get('normalize_mean', [0.5, 0.5, 0.5]),
            normalize_std=config.get('normalize_std', [0.5, 0.5, 0.5]),
            clip_samples=config.get('clip_samples', None)
        )
    else:
        # Create waveform preprocessor
        preprocessor = WaveformPreprocessor(
            target_sr=config['sample_rate'],
            target_length=config['target_length']
        )
    
    return preprocessor

