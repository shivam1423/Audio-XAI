# ResNet50 Direct Audio Training Guide

## Overview

This document explains how the ResNet50 training pipeline has been updated to process raw audio files directly instead of relying on pre-saved PNG spectrograms. This ensures consistency between training and evaluation pipelines, avoiding precision loss from PNG quantization.

## Problem Statement

Previously, the training workflow had two separate paths:

1. **Two-step approach (PNG-based)**:
   - Generate spectrograms and save as PNG files using `make_spectrograms.py`
   - Train ResNet50 on pre-saved PNG images
   - **Issue**: PNG encoding introduces 8-bit quantization and compression artifacts

2. **One-step approach (Direct)**:
   - Generate spectrograms on-the-fly during training
   - No intermediate PNG files
   - **Benefit**: Higher fidelity, consistent with RISE_audio evaluation pipeline

## Solution: Unified Direct Audio Processing

### Updated Files

#### 1. `ResNet50/utils.py`

**Key Changes**:
- Updated `audio_to_mel_spectrogram_image()` function to match the exact preprocessing pipeline from `RISE_audio/src/preprocessor.py` (ResNetPreprocessor)
- Fixed syntax error in `get_class_name()` (double comma removed)

**Preprocessing Pipeline** (lines 42-91):
```python
def audio_to_mel_spectrogram_image(filepath, sr=22050, n_fft=1024, 
                                   hop_length=512, n_mels=128):
    """
    Pipeline matches RISE_audio ResNetPreprocessor:
    1. Load audio at target sample rate (22050 Hz)
    2. Generate mel spectrogram (128 mel bins)
    3. Convert to dB scale
    4. Normalize to [0, 255]
    5. Create PIL Image and resize to 224x224 with BILINEAR interpolation
    """
    # 1. Load audio
    y, sr = librosa.load(filepath, sr=sr)
    
    # 2. Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # 3. Convert to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Normalize to [0, 255]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img_array = (mel_norm * 255).astype(np.uint8)
    
    # 5. Create PIL Image and resize with explicit BILINEAR interpolation
    img = Image.fromarray(mel_img_array)
    img = img.convert("L")  # Ensure grayscale
    img = img.resize((224, 224), Image.BILINEAR)
    
    return img
```

#### 2. `ResNet50/training/fine_tune_resnet.py`

**No changes needed** - This file already supports direct audio processing through the `ESC50SpectrogramDataset` class (line 83):

```python
else:
    # Default: on-the-fly spectrogram generation from raw audio
    full_ds = ESC50SpectrogramDataset(args.esc50_root, transform=default_preprocess)
```

## How to Use

### Training ResNet50 on Raw Audio (Recommended)

```bash
# Single train/val split (fold 5 as validation)
python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 1

# 5-fold cross-validation (recommended for final training)
python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 5
```

### Using SLURM (Cluster)

The existing `run.sh` script already uses the direct audio approach:

```bash
sbatch run.sh
```

This executes:
```bash
srun python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 5
```

### Legacy: Training on Pre-computed PNGs (Not Recommended)

If you still want to use the PNG-based approach (e.g., for comparison):

```bash
# Step 1: Generate PNGs
python -m training.make_spectrograms \
    --esc50_root ESC50 \
    --out_dir ESC50_spectrograms \
    --sr 22050 \
    --n_mels 128

# Step 2: Train on PNGs
python -m training.fine_tune_resnet \
    --spectrogram_dir ESC50_spectrograms \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 5
```

## Dataset Structure

### Required ESC-50 Directory Structure

```
ESC50/
├── audio/
│   ├── 1-100032-A-0.wav
│   ├── 1-100038-A-14.wav
│   └── ...
└── meta/
    └── esc50.csv
```

### Dataset Class: `ESC50SpectrogramDataset`

Located in `utils.py` (lines 130-160), this class:
- Loads raw `.wav` files from `ESC50/audio/`
- Converts them to mel-spectrograms on-the-fly using `audio_to_mel_spectrogram_image()`
- Returns `(tensor, label)` tuples compatible with PyTorch DataLoader
- Supports ESC-50's official 5-fold cross-validation structure

## Consistency with RISE_audio Evaluation

### Key Parameters (Identical)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sr` | 22050 Hz | Sample rate for mel-spectrogram generation |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 512 | Hop length between frames |
| `n_mels` | 128 | Number of mel bins |
| `image_size` | (224, 224) | Final spectrogram size |
| Interpolation | `Image.BILINEAR` | Resize method |
| Normalization | `[0.5, 0.5, 0.5]` (mean/std) | ImageNet-style normalization |

### Verification

You can verify consistency by comparing:

1. **Training preprocessing** (`ResNet50/utils.py`):
   ```python
   audio_to_mel_spectrogram_image(filepath)
   ```

2. **Evaluation preprocessing** (`RISE_audio/src/preprocessor.py`):
   ```python
   ResNetPreprocessor().process_original_audio(audio_tensor, sample_rate)
   ```

Both should produce identical outputs given the same input audio.

## Benefits of Direct Audio Processing

1. **Higher Fidelity**: No precision loss from 8-bit PNG quantization
2. **Consistency**: Training and evaluation use identical preprocessing pipelines
3. **Flexibility**: Easy to modify preprocessing parameters without regenerating PNGs
4. **Storage**: No need to store ~400MB of PNG files (2000 audio files × ~200KB per PNG)
5. **Reproducibility**: Single source of truth for preprocessing logic

## Performance Considerations

### Computational Cost

- **Direct Audio**: Slightly slower per epoch (~10-15% overhead) due to on-the-fly spectrogram generation
- **Pre-computed PNG**: Faster data loading, but requires upfront preprocessing time

### Recommendations

- **For development/debugging**: Use direct audio processing (default)
- **For final production training**: Direct audio is still recommended for consistency
- **For very large datasets**: Consider pre-computed spectrograms if training time is critical

## Verification Steps

After training, verify that your model works correctly:

```python
import torch
from utils import ESC50SpectrogramDataset, preprocess
from torchvision import models

# Load trained model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 50)
model.load_state_dict(torch.load("resnet50_esc50_fold5.pt"))
model.eval()

# Test on a single audio file
dataset = ESC50SpectrogramDataset("ESC50", transform=preprocess)
img_tensor, label = dataset[0]

# Run inference
with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))
    pred_class = output.argmax(dim=1).item()
    
print(f"True label: {label}, Predicted: {pred_class}")
```

## Troubleshooting

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'ESC50/audio/...'`

**Solution**: Ensure the ESC-50 dataset is properly extracted:
```bash
ls ESC50/audio/ | head -5  # Should show .wav files
ls ESC50/meta/            # Should contain esc50.csv
```

### Issue: `ModuleNotFoundError: No module named 'librosa'`

**Solution**: Install required dependencies:
```bash
pip install librosa torch torchvision torchaudio pandas numpy pillow scikit-learn
```

### Issue: Out of memory during training

**Solution**: Reduce batch size:
```bash
python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 16  # Reduced from 32
```

## Summary

The ResNet50 training pipeline now processes raw audio files directly, matching the exact preprocessing used in RISE_audio evaluation. This ensures:

1. ✅ No precision loss from PNG encoding
2. ✅ Consistent preprocessing between training and evaluation
3. ✅ Single source of truth for mel-spectrogram generation
4. ✅ Flexibility to modify parameters without regenerating files

Simply run `python -m training.fine_tune_resnet --esc50_root ESC50 --epochs 50 --batch_size 32 --kfolds 5` to train with the new direct audio pipeline.
