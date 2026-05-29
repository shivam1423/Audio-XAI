# Quick Start Guide - ResNet50 UrbanSound8K

This guide will help you get started with training ResNet50 on UrbanSound8K in just a few steps.

## Prerequisites

✓ Python 3.7+
✓ UrbanSound8K dataset downloaded
✓ GPU with CUDA (recommended, but optional)

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd Resnet50_UrbanSound8K
pip install -r requirements.txt
```

### 2. Verify Dataset

Make sure your UrbanSound8K dataset is located at `../UrbanSound8K` (relative to this directory) or update the path in commands below.

Expected structure:
```
../UrbanSound8K/
├── fold1/, fold2/, ..., fold10/
└── UrbanSound8K.csv
```

## Training (Quick Start)

### Option A: Using SLURM (Cluster)

```bash
# Edit the DATA_DIR in scripts/run_train.sh if needed
# Then submit:
sbatch scripts/run_train.sh
```

Monitor progress:
```bash
tail -f slurm-*.out
```

### Option B: Local Training (GPU)

```bash
python scripts/train.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./trained_models \
    --device cuda \
    --batch_size 32 \
    --epochs 100
```

### Option C: Local Training (CPU)

```bash
python scripts/train.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./trained_models \
    --device cpu \
    --batch_size 16 \
    --epochs 50
```

## Evaluation

After training completes, evaluate the best model:

```bash
python scripts/evaluate.py \
    --data_dir ../UrbanSound8K \
    --model_path ./trained_models/best_model.pth \
    --output_dir ./results
```

## Expected Outputs

### Training Outputs (in `trained_models/`)
- `best_model.pth` - Best model checkpoint
- `training_log_*.csv` - Training history
- `checkpoint_epoch_*.pth` - Periodic checkpoints

### Evaluation Outputs (in `results/`)
- `evaluation_results.json` - All metrics
- `confusion_matrix.png` - Confusion matrix plot
- `per_class_metrics.png` - Per-class performance
- `predictions.npy` - Model predictions
- `labels.npy` - True labels

## Expected Performance

- **Target Accuracy**: 78-84% on test fold
- **Training Time**: 
  - GPU (single GTX 1080): ~4-6 hours
  - CPU: ~24-48 hours
- **Convergence**: Usually by epoch 50-70

## Configuration Overview

Default settings (can be modified via command-line arguments):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Training Folds | 1-8 | ~7,000 samples |
| Validation Fold | 9 | ~800 samples |
| Test Fold | 10 | ~800 samples |
| Sample Rate | 22050 Hz | Audio resampling |
| Spectrogram Size | 128x128 | Mel spectrogram dimensions |
| Batch Size | 32 | Training batch size |
| Epochs | 100 | Total training epochs |
| Learning Rate | 0.001 | Initial LR |
| Optimizer | SGD | With momentum 0.9 |

## Troubleshooting

**Out of Memory Error?**
```bash
# Reduce batch size
python scripts/train.py --data_dir ../UrbanSound8K --batch_size 16
```

**Dataset Not Found?**
```bash
# Verify path
ls ../UrbanSound8K/
# Should see: fold1/, fold2/, ..., fold10/, UrbanSound8K.csv
```

**Slow Training?**
```bash
# Increase workers (if you have multiple CPU cores)
python scripts/train.py --data_dir ../UrbanSound8K --num_workers 8
```

## Next Steps

1. **Monitor Training**: Watch the training log CSV file or console output
2. **Evaluate Model**: Run evaluation script on test fold
3. **Analyze Results**: Check confusion matrix to see which classes are confused
4. **Iterate**: Adjust hyperparameters if needed (learning rate, batch size, etc.)

## File Structure Overview

```
Resnet50_UrbanSound8K/
├── config/          # Configuration settings
├── data/            # Dataset loader and preprocessing
├── models/          # ResNet50 architecture
├── training/        # Training loop and trainer
├── evaluation/      # Evaluation and metrics
├── scripts/         # Main training and evaluation scripts
├── utils/           # Helper utilities
├── trained_models/  # Model checkpoints (created during training)
├── results/         # Evaluation results (created during evaluation)
├── requirements.txt # Python dependencies
├── README.md        # Full documentation
└── START_HERE.md    # This file
```

## Full Documentation

For detailed documentation, see [README.md](README.md)

## Support

If you encounter issues:
1. Check the [Troubleshooting section in README.md](README.md#troubleshooting)
2. Verify dataset structure
3. Check Python and PyTorch versions
4. Ensure CUDA is properly installed (for GPU training)

---

**Happy Training! 🚀**
