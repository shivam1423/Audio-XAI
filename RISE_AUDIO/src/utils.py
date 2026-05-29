#!/usr/bin/env python
# coding: utf-8
import os

# Dataset configuration - Change this to switch datasets!
# Options: 'esc50', 'urbansound8k'
DEFAULT_DATASET = os.environ.get('DATASET', 'esc50')

# Model type configuration - Change this to switch models!
# Options: 'resnet', 'wav2vec2', 'htsat', 'acdnet'
DEFAULT_MODEL_TYPE = os.environ.get('MODEL_TYPE')

DEFAULT_INPUT_DIR = os.environ.get('INPUT_DIR', '../ESC50/audio_esc10')
DEFAULT_OUTPUT_DIR = os.environ.get('OUTPUT_DIR')
MASKS_DIR = 'results/masks'

"""Default settings and configurations for TF-Structured RISE."""
# Audio config
TARGET_ORIGINAL_SR = 48000
original_nfft = 3072

# Audio → Mel parameters
MEL_SR = 22050
MEL_N_FFT = 1024
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
MEL_IMAGE_SIZE = (224, 224)

# Model settings
MODEL_WEIGHTS_PATH = 'resnet50_esc50.pt'
MODEL_INPUT_SIZE = (224, 224)

# Dataset-specific configurations
DATASET_CONFIG = {
    'esc50': {
        'num_classes': 50,
        'audio_dir': '../ESC50/audio',
        'model_paths': {
            'resnet': 'checkpoints/resnet50_esc50.pt',
            'wav2vec2': 'checkpoints/best_model_wav2vec2.pt',
            'htsat': 'checkpoints/HTSAT.ckpt',
            'acdnet': 'checkpoints/acdnet_weight_pruned_trained_fold4_90.50.pt'
        }
    },
    'urbansound8k': {
        'num_classes': 10,
        'audio_dir': '../UrbanSound8K/audio/fold10',
        'model_paths': {
            'resnet': 'checkpoints/resnet50_urbansound8k.pt',
            'wav2vec2': 'checkpoints/best_model_wav2vec2_us8k.pt',
            'htsat': '../HTSAT_UrbanSound8K/training_output_fold10/best_model.pth',
            'acdnet': 'checkpoints/acdnet_us8k_best.pt'
        }
    }
}

# Get current dataset config
NUM_CLASSES = DATASET_CONFIG[DEFAULT_DATASET]['num_classes']

# Model-specific checkpoint paths (uses dataset-aware paths)
def get_model_path(model_type, dataset=None):
    """Get model path for specific model type and dataset."""
    if dataset is None:
        dataset = DEFAULT_DATASET
    return DATASET_CONFIG[dataset]['model_paths'].get(model_type)

MODEL_PATHS = DATASET_CONFIG[DEFAULT_DATASET]['model_paths']

# Model-specific sample rates for feature extraction
MODEL_SAMPLE_RATES = {
    'resnet': 22050,    # Mel spectrogram generation at 22050 Hz
    'wav2vec2': 16000,  # wav2vec2 expects 16kHz raw audio
    'htsat': 32000,     # HTSAT uses 32kHz raw audio
    'acdnet': 20000     # ACDNet uses 20kHz raw audio
}

# Mask generation defaults
DEFAULT_N_MASKS            = 6000
DEFAULT_TIME_STRIPE_FRAC   = 0.25
DEFAULT_FREQ_BAND_FRAC     = 0.25
DEFAULT_RECT_PATCH_FRAC    = 0.25
DEFAULT_MEL_BAND_FRAC      = 0.25
DEFAULT_SINGLE_BLOCK       = False

# Mask parameters
TIME_STRIPE_WIDTH_PX       = (4, 24)
FREQ_BAND_HEIGHT_PX        = (4, 24)
RECT_SIZE_PX               = (8, 48)
RECT_COUNT_RANGE           = (1, 6)
STRIPE_COUNT_RANGE         = (1, 12)
MEL_BANDS                  = 64
BAND_KEEP_PROB             = 0.3

# TIME_STRIPE_WIDTH_PX = (6, 32)      # Increased from (4, 24)
# FREQ_BAND_HEIGHT_PX = (6, 32)       # Increased from (4, 24)
# RECT_SIZE_PX = (12, 64)             # Increased from (8, 48)
# RECT_COUNT_RANGE = (1, 6)
# STRIPE_COUNT_RANGE = (1, 12)
# MEL_BANDS = 64                      # Keep same
# BAND_KEEP_PROB = 0.4               # Increased from 0.3

# Soft masking defaults
DEFAULT_SOFT_MASKING       = os.environ.get('SOFT_MASKING')
DEFAULT_EDGE_SIGMA_PX      = 1.0

# Occlusion defaults
DEFAULT_OCCLUSION          = os.environ.get('OCCLUSION')
# GPU settings
MODEL_GPU_BATCH = {
    'resnet': 150,      # ResNet can handle larger batches (mel specs are small)
    'wav2vec2': 64,     # Wav2Vec2 needs much smaller batches (raw audio is large)
    'htsat': 32,        # HTSAT uses transformer with 10s audio (320k samples), needs small batch
    'acdnet': 150
}

# Misc.
MASK_PROGRESS_PRINT_EVERY  = 2000
AUDIO_GLOB_PATTERN         = '*.wav'