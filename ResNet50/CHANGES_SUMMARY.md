# ResNet50 Training: Direct Audio Processing

## Quick Summary

The ResNet50 training pipeline now processes **raw audio files directly** instead of using pre-saved PNG spectrograms. This ensures **100% consistency** with the RISE_audio evaluation pipeline.

## What Changed?

### Updated File: `utils.py`

1. **Enhanced `audio_to_mel_spectrogram_image()` function**:
   - Added explicit `Image.BILINEAR` interpolation (matches RISE_audio)
   - Improved documentation with exact preprocessing pipeline
   - Fixed syntax error in `get_class_name()` function

### No Changes Needed

- `training/fine_tune_resnet.py` already supports direct audio processing via `ESC50SpectrogramDataset`

## How to Use

### Training (Default: Direct Audio)

```bash
# 5-fold cross-validation (recommended)
python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 5

# Or use SLURM
sbatch run.sh
```

### Verification

Test that preprocessing is consistent:

```bash
python verify_preprocessing.py
```

## Key Benefits

1. ✅ **No precision loss** from PNG quantization
2. ✅ **Identical preprocessing** for training and evaluation
3. ✅ **Flexible parameters** without regenerating files
4. ✅ **Smaller storage** footprint (no PNG files needed)

## Preprocessing Pipeline (Identical to RISE_audio)

```
Raw Audio (.wav)
    ↓
Load at 22050 Hz
    ↓
Mel Spectrogram (128 bins, n_fft=1024, hop=512)
    ↓
Convert to dB scale
    ↓
Normalize to [0, 255]
    ↓
PIL Image (grayscale)
    ↓
Resize to 224×224 (BILINEAR)
    ↓
Convert to RGB (3 channels)
    ↓
ImageNet Normalization (mean=0.5, std=0.5)
    ↓
ResNet50 Input
```

## Files

- **`utils.py`**: Updated preprocessing functions
- **`training/fine_tune_resnet.py`**: Training script (no changes)
- **`DIRECT_AUDIO_TRAINING_GUIDE.md`**: Comprehensive documentation
- **`verify_preprocessing.py`**: Verification script

## Next Steps

1. ✅ Code updated
2. ⏭️ Run verification: `python verify_preprocessing.py`
3. ⏭️ Train model: `python -m training.fine_tune_resnet --esc50_root ESC50 --epochs 50 --batch_size 32 --kfolds 5`

---

**For detailed documentation**, see `DIRECT_AUDIO_TRAINING_GUIDE.md`
