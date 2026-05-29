# Quick Start Guide: HTSAT UrbanSound8K Evaluation

## Prerequisites

1. **AudioSet Checkpoint**: Download HTSAT AudioSet checkpoint (e.g., from official HTSAT repository)
2. **UrbanSound8K Dataset**: Download and extract UrbanSound8K dataset
3. **Dependencies**: Install required packages

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

Ensure UrbanSound8K is organized as:
```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

## Step 3: Download AudioSet Checkpoint

Place the AudioSet checkpoint file in this directory or note its path.

Example checkpoint name: `HTSAT_AudioSet.ckpt`

## Step 4: Run Evaluation

### Option A: Command Line

```bash
python evaluate_urbansound8k.py \
    --checkpoint /path/to/HTSAT_AudioSet.ckpt \
    --audio_dir /path/to/UrbanSound8K/audio \
    --metadata /path/to/UrbanSound8K/metadata/UrbanSound8K.csv \
    --test_fold 10 \
    --batch_size 32 \
    --device cuda \
    --output_dir ./results
```

### Option B: Shell Script

1. Edit `run_evaluation.sh` and update paths:
   - `CHECKPOINT`: Path to AudioSet checkpoint
   - `AUDIO_DIR`: Path to UrbanSound8K audio directory
   - `METADATA`: Path to UrbanSound8K.csv

2. Run:
```bash
./run_evaluation.sh
```

## Step 5: View Results

Results will be saved in the output directory (default: `./results`):

- `predictions.csv`: Per-sample predictions
- `confusion_matrix.csv`: Confusion matrix
- `summary.txt`: Evaluation summary

## Expected Output

You should see:
- Overall accuracy (expected ~77% for zero-shot)
- Per-class metrics
- Confusion matrix

## Troubleshooting

### "HTS-Audio-Transformer not found"
The code expects the official HTSAT repository at `../HTSAT/HTS-Audio-Transformer/`. 
If it's elsewhere, create a symlink or modify `load_audioset_model.py`.

### "CUDA out of memory"
Reduce batch size:
```bash
--batch_size 16
```

### "Metadata file not found"
Provide explicit path:
```bash
--metadata /full/path/to/UrbanSound8K.csv
```

## Next Steps

- Try different test folds (1-10)
- Evaluate on all folds for comprehensive results
- Fine-tune the model for better performance

