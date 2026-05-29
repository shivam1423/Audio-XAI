# HTSAT Evaluation - Complete Usage Guide

This guide provides multiple methods to evaluate the HTSAT model on ESC-50 dataset.

## Quick Start (Recommended)

### Method 1: Using Simplified Implementation (Fastest)

This uses a custom implementation designed for evaluation:

```bash
# 1. Test the setup
python test_setup.py

# 2. Run evaluation
./run_evaluation.sh
```

Or with Python directly:
```bash
python evaluate.py --val_fold 2 --device cuda
```

### Method 2: Using Official HTSAT Repository

For the most authentic results using the original codebase:

```bash
# 1. Clone and setup official repository
./setup_official_htsat.sh

# 2. Run evaluation with official code
python evaluate_with_official_repo.py --val_fold 2
```

## Detailed Instructions

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- TorchAudio >= 2.0.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)
- Librosa, tqdm

### Configuration

The evaluation is configured for:
- **Validation fold**: 2
- **Training folds**: 1, 3, 4, 5
- **Checkpoint**: `HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt`
- **Audio directory**: `/Users/shivampandey/SS 25/Thesis/RISE_dev/ESC50/audio`
- **Sample rate**: 32kHz
- **Clip length**: 10 seconds

Edit `htsat_config.py` to change these settings.

### Step-by-Step: Simplified Implementation

#### Step 1: Verify Setup
```bash
python test_setup.py
```

This checks:
- All dependencies are installed
- Checkpoint file exists and is readable
- Audio files are accessible
- PyTorch/CUDA setup is correct
- Dataset loading works
- Model can be created and loaded
- Forward pass works

#### Step 2: Run Evaluation
```bash
# Using shell script (easiest)
./run_evaluation.sh

# Or using Python directly
python evaluate.py \
    --checkpoint ./HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt \
    --audio_dir ../ESC50/audio \
    --val_fold 2 \
    --batch_size 32 \
    --device cuda \
    --output_dir ./results
```

#### Step 3: Review Results

Results are saved in the `results/` (or specified output) directory:

1. **predictions.csv**: Per-sample predictions
2. **confusion_matrix.npy**: Confusion matrix array
3. **confusion_matrix.png**: Visualization
4. **evaluation_summary.txt**: Summary statistics

### Step-by-Step: Official Repository

#### Step 1: Setup Official Repository
```bash
./setup_official_htsat.sh
```

This will:
- Clone the official HTSAT repository
- Install dependencies
- Setup the environment

#### Step 2: Run Evaluation
```bash
python evaluate_with_official_repo.py \
    --checkpoint ./HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt \
    --audio_dir ../ESC50/audio \
    --val_fold 2 \
    --device cuda \
    --output_dir ./results_official
```

## Command-Line Options

### For `evaluate.py`

```
--checkpoint PATH       Path to HTSAT checkpoint (.ckpt file)
--audio_dir PATH        Path to ESC-50 audio directory
--metadata PATH         Path to esc50.csv (optional, auto-detected)
--val_fold INT          Validation fold number (default: 2)
--batch_size INT        Batch size for evaluation (default: 32)
--num_workers INT       Data loading workers (default: 4)
--device STR            Device: 'cuda' or 'cpu' (default: 'cuda')
--output_dir PATH       Output directory for results (default: './results')
```

### Examples

Evaluate on fold 2 with GPU:
```bash
python evaluate.py --val_fold 2 --device cuda
```

Evaluate on fold 5 with CPU (no GPU):
```bash
python evaluate.py --val_fold 5 --device cpu --batch_size 16
```

Evaluate with custom paths:
```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio_dir /path/to/audio \
    --val_fold 2 \
    --output_dir /path/to/results
```

## Understanding the Output

### Console Output

During evaluation, you'll see:
1. Model loading information
2. Dataset statistics
3. Progress bar during evaluation
4. Overall accuracy
5. Per-class precision, recall, F1-score

### Files Generated

#### 1. predictions.csv
```csv
filename,true_label,predicted_label,correct
1-100032-A-0.wav,0,0,True
2-100210-A-1.wav,1,1,True
...
```

#### 2. confusion_matrix.npy
NumPy array of shape (50, 50) containing confusion matrix.

Load with:
```python
import numpy as np
cm = np.load('results/confusion_matrix.npy')
```

#### 3. confusion_matrix.png
Heatmap visualization of the confusion matrix showing true vs predicted labels.

#### 4. evaluation_summary.txt
Text file with:
- Configuration used
- Dataset statistics
- Overall accuracy
- Timestamp

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python evaluate.py --batch_size 16
```

**Solution 2**: Use CPU
```bash
python evaluate.py --device cpu
```

### Issue: Checkpoint Loading Error

**Symptom**: `RuntimeError: Error loading checkpoint`

**Solutions**:
1. Check checkpoint path is correct
2. Verify checkpoint file is not corrupted
3. Ensure checkpoint was trained for ESC-50 (50 classes)
4. Try loading with strict=False (code does this automatically)

### Issue: Audio Files Not Found

**Symptom**: `Error: Audio directory not found`

**Solutions**:
1. Verify audio directory path in `htsat_config.py`
2. Check audio files exist and are `.wav` format
3. Ensure proper file permissions

### Issue: Metadata File Missing

**Symptom**: `Warning: Metadata file not found`

**Solution**: The code automatically creates metadata from filenames. This is normal if you don't have `esc50.csv`. The filenames must follow ESC-50 format:
```
{fold}-{clip_id}-{take}-{target}.wav
Example: 1-100032-A-0.wav
```

### Issue: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Slow Evaluation

**Solutions**:
1. Use GPU: `--device cuda`
2. Increase workers: `--num_workers 8`
3. Increase batch size: `--batch_size 64`

## Performance Optimization

### For Faster Evaluation

1. **Use GPU**:
   ```bash
   python evaluate.py --device cuda --batch_size 64
   ```

2. **Increase workers**:
   ```bash
   python evaluate.py --num_workers 8
   ```

3. **Mixed precision** (modify code to use `torch.cuda.amp`)

### For Limited Memory

1. **Reduce batch size**:
   ```bash
   python evaluate.py --batch_size 8
   ```

2. **Reduce workers**:
   ```bash
   python evaluate.py --num_workers 2
   ```

3. **Use CPU**:
   ```bash
   python evaluate.py --device cpu
   ```

## Advanced Usage

### Evaluating Multiple Folds

Create a script to evaluate all folds:

```bash
#!/bin/bash
for fold in 1 2 3 4 5; do
    echo "Evaluating fold $fold..."
    python evaluate.py \
        --val_fold $fold \
        --output_dir ./results_fold${fold}
done
```

### Custom Post-Processing

Load and analyze results in Python:

```python
import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('results/predictions.csv')

# Calculate per-fold accuracy
accuracy = df['correct'].mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Load confusion matrix
cm = np.load('results/confusion_matrix.npy')

# Find most confused classes
import numpy as np
# Set diagonal to 0 to find off-diagonal max
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
confused_pair = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
print(f"Most confused pair: {confused_pair}")
```

### Batch Processing

Evaluate multiple checkpoints:

```python
import os
import subprocess

checkpoints = [
    'checkpoint_fold1.ckpt',
    'checkpoint_fold2.ckpt',
    # ...
]

for i, ckpt in enumerate(checkpoints):
    cmd = f"python evaluate.py --checkpoint {ckpt} --output_dir results_ckpt{i}"
    subprocess.run(cmd, shell=True)
```

## Citation

If you use HTSAT in your research:

```bibtex
@inproceedings{chen2022hts,
  title={HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  author={Chen, Ke and Du, Xingjian and Zhu, Bilei and Ma, Zejun and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={ICASSP 2022},
  pages={646--650},
  year={2022},
  organization={IEEE}
}
```

## Support

For issues:
1. Check this guide
2. Run `python test_setup.py` to diagnose
3. Review the official repository: https://github.com/RetroCirce/HTS-Audio-Transformer
4. Check PyTorch/CUDA compatibility

## Files Reference

| File | Purpose |
|------|---------|
| `htsat_config.py` | Configuration settings |
| `htsat_model.py` | HTSAT model implementation |
| `esc50_dataset.py` | ESC-50 dataset loader |
| `evaluate.py` | Main evaluation script |
| `test_setup.py` | Setup verification script |
| `run_evaluation.sh` | Bash script for evaluation |
| `setup_official_htsat.sh` | Setup official repository |
| `evaluate_with_official_repo.py` | Evaluation using official code |
| `requirements.txt` | Python dependencies |
| `README.md` | Quick start guide |
| `USAGE_GUIDE.md` | This file |




