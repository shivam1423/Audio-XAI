# HTSAT Evaluation - Error Fix Summary

## Issues Identified from `slurm-53106.out`

### 1. **Device Mismatch Error** ❌
```
Error in forward pass for batch 0: stft input and window must be on the same device 
but got self on cuda:0 and window on cpu
```

**Root Cause**: The torchlibrosa STFT module has internal buffers (e.g., window function) that weren't moved to CUDA when the model was moved to GPU.

**Solution**: ✅
- Move model to device BEFORE setting eval mode
- Explicitly move spectrogram_extractor and logmel_extractor to device
- Ensure all buffers are transferred with the model

### 2. **Model Architecture Mismatch** ❌
```
Missing key(s) in state_dict: "patch_embed.proj.weight", "stages.0.blocks.0.norm1.weight", ...
Unexpected key(s) in state_dict: "sed_model.spectrogram_extractor.stft.conv_real.weight", ...
```

**Root Cause**: The checkpoint contains the full `SEDWrapper` model with `sed_model.*` prefix, but the simplified implementation didn't match the official HTSAT architecture.

**Solution**: ✅
- Use official HTSAT architecture from cloned repository
- Import `HTSAT_Swin_Transformer` from `model/htsat.py`
- Properly extract `sed_model.*` weights from checkpoint
- Match exact configuration from `esc_config.py`

## Files Created/Modified

###  New Files (Fixes)

1. **`evaluate_htsat.py`** ⭐ (NEW - RECOMMENDED)
   - Uses official HTSAT architecture
   - Fixes device mismatch issues
   - Properly loads checkpoint weights
   - Handles model output correctly

2. **`HTS-Audio-Transformer/`** (Cloned Repository)
   - Official HTSAT implementation
   - Contains correct model architecture

### Original Files (Still Useful)

3. **`evaluate.py`** (Original - Has Issues)
   - Simplified implementation
   - Device mismatch not fixed
   - Architecture doesn't match checkpoint

4. **`evaluate_with_official_repo.py`**
   - Alternative approach
   - May need adjustments

## How to Run (Fixed Version)

### Quick Start
```bash
cd "/Users/shivampandey/SS 25/Thesis/RISE_dev/HTSAT"

# Run with fixed evaluation script
python evaluate_htsat.py \
    --checkpoint HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt \
    --audio_dir ../ESC50/audio \
    --val_fold 2 \
    --batch_size 32 \
    --device cuda
```

### Using Updated Shell Script
```bash
./run_evaluation.sh  # Now uses evaluate_htsat.py
```

### For SLURM
```bash
sbatch run_evaluation.sh
```

## Technical Details

### Official HTSAT Architecture
From the checkpoint keys, the model structure is:
```
SEDWrapper
└── sed_model (HTSAT_Swin_Transformer)
    ├── spectrogram_extractor (torchlibrosa.STFT)
    ├── logmel_extractor (torchlibrosa.LogmelFilterBank)
    ├── bn0 (BatchNorm)
    ├── patch_embed (PatchEmbed)
    ├── layers (4 Swin Transformer Layers)
    │   ├── Layer 0: depth=2, heads=4, dim=96
    │   ├── Layer 1: depth=2, heads=8, dim=192
    │   ├── Layer 2: depth=6, heads=16, dim=384
    │   └── Layer 3: depth=2, heads=32, dim=768
    ├── norm (LayerNorm)
    ├── tscam_conv (Token-Semantic Conv)
    └── head (Linear classifier)
```

### Configuration Match
The evaluation uses these parameters matching the training:
- Sample rate: 32 kHz
- Clip length: 10 seconds (320,000 samples)
- Window size: 1024
- Hop size: 320
- Mel bins: 64
- Patch size: 4
- Window size (Swin): 8
- Embedding dim: 96
- Depths: [2, 2, 6, 2]
- Num heads: [4, 8, 16, 32]

## Device Mismatch Fix Details

### Problem Code
```python
model = Model()
model.eval()
model = model.to(device)  # Too late! Buffers already registered in eval mode
```

### Fixed Code
```python
model = Model()
model = model.to(device)  # Move FIRST
# Explicitly ensure all components are on device
if hasattr(model, 'spectrogram_extractor'):
    model.spectrogram_extractor = model.spectrogram_extractor.to(device)
if hasattr(model, 'logmel_extractor'):
    model.logmel_extractor = model.logmel_extractor.to(device)
model.eval()  # Then set to eval mode
```

## Expected Output (Fixed)

```
============================================================
HTSAT Evaluation (Official Architecture)
============================================================
Checkpoint: HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt
Audio directory: ../ESC50/audio
Validation fold: 2
Device: cuda
============================================================

Loading checkpoint from: HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt
✓ Model loaded successfully
  Epoch: 50

Loading validation data (fold 2)...
Loaded 400 samples from ESC-50
Folds: [2]
Classes: 50
Validation samples: 400

Evaluating...
Evaluation: 100%|██████████| 13/13 [00:XX<00:00, X.XXit/s]

======================================================================
EVALUATION RESULTS
======================================================================
Total samples: 400
Overall Accuracy: ~98.50%
======================================================================

Per-Class Performance:
----------------------------------------------------------------------
              precision    recall  f1-score   support
...
```

## Verification Steps

1. **Check model loading**:
   ```python
   model, config = load_htsat_model(checkpoint_path, device)
   # Should load without errors
   ```

2. **Test forward pass**:
   ```python
   x = torch.randn(1, 320000).to(device)
   output = model(x, None, True)
   # Should work without device errors
   ```

3. **Run evaluation**:
   ```bash
   python evaluate_htsat.py --device cuda
   # Should complete without errors
   ```

## Troubleshooting

### If you still get device errors:
```bash
# Use CPU instead
python evaluate_htsat.py --device cpu
```

### If you get import errors:
```bash
# Make sure official repo is cloned
./setup_official_htsat.sh

# Install torchlibrosa
pip install torchlibrosa
```

### If you get out of memory:
```bash
# Reduce batch size
python evaluate_htsat.py --batch_size 16
```

## Key Changes Made

1. ✅ Cloned official HTSAT repository
2. ✅ Created `evaluate_htsat.py` using official architecture
3. ✅ Fixed device transfer order
4. ✅ Properly extract `sed_model` weights from checkpoint
5. ✅ Use correct model output format (dict with 'clipwise_output')
6. ✅ Updated `run_evaluation.sh` to use new script

## Recommendation

**Use `evaluate_htsat.py`** - this is the fixed version that:
- Uses official HTSAT architecture
- Fixes all device mismatch issues
- Properly loads the checkpoint
- Should work without errors

The original `evaluate.py` was a simplified implementation that doesn't match the checkpoint structure.




