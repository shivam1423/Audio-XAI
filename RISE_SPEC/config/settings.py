#!/usr/bin/env python
# coding: utf-8

"""Default settings and configurations for TF-Structured RISE."""

# Model settings
MODEL_WEIGHTS_PATH = 'resnet50_esc50.pt'
MODEL_INPUT_SIZE = (224, 224)
NUM_CLASSES = 50

# Data settings
DEFAULT_DATADIR = 'ESC50_spectrograms/'
DEFAULT_AUDIO_DIR = 'test_audio/'  # Relative path for cluster compatibility
DEFAULT_WORKERS = 2

# Audio file extensions
AUDIO_EXTENSIONS = ('.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.ogg', '.OGG')

# Mask generation defaults
DEFAULT_N_MASKS = 6000
DEFAULT_TIME_STRIPE_FRAC = 0.25
DEFAULT_FREQ_BAND_FRAC = 0.25
DEFAULT_RECT_PATCH_FRAC = 0.25
DEFAULT_MEL_BAND_FRAC = 0.25

# Mask parameters
TIME_STRIPE_WIDTH_PX = (4, 24)
FREQ_BAND_HEIGHT_PX = (4, 24)
RECT_SIZE_PX = (8, 48)
RECT_COUNT_RANGE = (1, 6)
STRIPE_COUNT_RANGE = (1, 12)
MEL_BANDS = 64
BAND_KEEP_PROB = 0.3

# Soft masking defaults
DEFAULT_SOFT_MASKING = "bilinear"
DEFAULT_EDGE_SIGMA_PX = 1.0

# Occlusion defaults
DEFAULT_OCCLUSION = "black"

# GPU settings
DEFAULT_GPU_BATCH = 250

# File paths
RESULTS_DIR = 'results'
MASKS_DIR = 'results/masks'
SALIENCY_DIR = 'results/saliency'

# Image extensions
IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF', '.TIFF'
)
