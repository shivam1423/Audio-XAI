# HTSAT UrbanSound8K Fine-tuning Guide

## Overview

This guide explains how to fine-tune HTSAT AudioSet checkpoint on UrbanSound8K for improved accuracy.

## Why Fine-tune?

- **Zero-shot (current)**: 7.89% accuracy (random 10-class head)
- **Fine-tuned (expected)**: 85-95% accuracy
- **Method**: Transfer learning - use AudioSet features, train new 10-class head

## Quick Start

### Option 1: Shell Script (Recommended)

```bash
# Edit run_training.sh to set paths
./run_training.sh
```

### Option 2: Python Command

```bash
python train_urbansound8k.py \
    --checkpoint HTSAT_AudioSet_Saved_1.ckpt \
    --audio_dir ../UrbanSound8K/audio \
    --metadata ../UrbanSound8K/metadata/UrbanSound8K.csv \
    --test_fold 10 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --freeze_features \
    --device cuda \
    --output_dir ./training_output
```

## Training Options

### Basic Configuration

- `--checkpoint`: Path to AudioSet checkpoint (required)
- `--audio_dir`: Path to UrbanSound8K audio directory (required)
- `--metadata`: Path to UrbanSound8K.csv (auto-searched if not provided)
- `--test_fold`: Test fold number (default: 10)

### Training Hyperparameters

- `--epochs`: Number of training epochs (default: 30)
  - Recommended: 20-50 epochs
  - More epochs = better accuracy but longer training

- `--batch_size`: Batch size (default: 16)
  - Larger = faster training but more GPU memory
  - GTX 1080 Ti (11GB): batch_size 16-32
  - Reduce if out of memory

- `--lr`: Learning rate (default: 1e-4)
  - Default works well for most cases
  - Try 1e-3 for faster convergence (may be less stable)
  - Try 1e-5 for more stable but slower training

### Advanced Options

- `--freeze_features`: Freeze AudioSet feature layers
  - **Recommended**: Faster training, good accuracy
  - Only trains classification head (~4M parameters)
  - Training time: ~30-60 minutes (30 epochs)

- Without `--freeze_features`: Full fine-tuning
  - Trains all layers (~89M parameters)
  - Better accuracy (1-2% improvement)
  - Training time: ~2-3 hours (30 epochs)

## Expected Training Time

| Configuration | Epochs | Time | Final Accuracy |
|--------------|--------|------|----------------|
| Freeze features | 30 | 30-60 min | 85-90% |
| Full fine-tuning | 30 | 2-3 hours | 88-95% |
| Freeze features | 50 | 1-1.5 hours | 88-92% |

*Times are for GTX 1080 Ti with batch_size=16*

## Output Files

Training saves to `--output_dir` (default: `./training_output`):

```
training_output/
├── best_model.pth              # Best checkpoint (highest val accuracy)
├── checkpoint_epoch_5.pth      # Checkpoint at epoch 5
├── checkpoint_epoch_10.pth     # Checkpoint at epoch 10
├── checkpoint_epoch_15.pth     # ...
├── training_history.csv        # Loss and accuracy per epoch
└── final_results.txt           # Final evaluation results
```

## Training Progress

You'll see output like:

```
Epoch 1/30
----------------------------------------------------------------------
Epoch 1 [Train]: 100%|████| 150/150 [00:45<00:00, 3.3it/s, loss=2.1234]
Epoch 1 [Val]:   100%|████| 27/27 [00:05<00:00, 5.2it/s, loss=1.8765]

Epoch 1 Results:
  Train Loss: 2.1234 | Train Acc: 35.42%
  Val Loss:   1.8765 | Val Acc:   42.18%
  LR: 0.000100
  ✓ Saved best model (val_acc: 42.18%)
```

## Using Trained Model

After training, evaluate with your best model:

```bash
python evaluate_urbansound8k_trained.py \
    --checkpoint ./training_output/best_model.pth \
    --audio_dir ../UrbanSound8K/audio \
    --test_fold 10
```

Or modify `evaluate_urbansound8k.py` to load from `.pth` instead of `.ckpt`.

## Tips for Better Results

### 1. Hyperparameter Tuning

Try different learning rates:
```bash
# Fast convergence
--lr 1e-3 --epochs 20

# Stable training
--lr 1e-4 --epochs 30

# Fine-grained optimization
--lr 1e-5 --epochs 50
```

### 2. Data Augmentation

Add to dataset loader:
- Time stretching
- Pitch shifting
- Noise addition
- Mixup

### 3. Cross-validation

Train on different folds:
```bash
# Test on fold 1, train on 2-10
--test_fold 1

# Test on fold 2, train on 1,3-10
--test_fold 2
```

Average results across all 10 folds for robust evaluation.

### 4. Ensemble

Train multiple models and average predictions:
- Different random seeds
- Different learning rates
- Different architectures

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch_size 8  # or even 4
```

### Slow Training

- Use `--freeze_features` flag
- Increase batch size if GPU allows
- Reduce `--num_workers` in config

### Low Accuracy

- Train for more epochs (50-100)
- Try without `--freeze_features` (full fine-tuning)
- Reduce learning rate (`--lr 1e-5`)
- Add data augmentation

### Overfitting

- Add dropout (modify model code)
- Reduce epochs
- Increase weight decay in optimizer
- Use data augmentation

## Expected Results by Epoch

Typical training progression with `--freeze_features`:

| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 35% | 42% | Initial learning |
| 5 | 75% | 68% | Rapid improvement |
| 10 | 88% | 78% | Approaching plateau |
| 20 | 95% | 85% | Near optimal |
| 30 | 97% | 87% | Slight overfitting |

## Comparison to Literature

Published UrbanSound8K results with AudioSet pre-training:
- HTSAT zero-shot: ~77% (using 527-class head)
- HTSAT fine-tuned: ~85-95%
- Your zero-shot (random head): 7.89%
- Your fine-tuned (expected): 85-95%

## Next Steps

After successful training:
1. Evaluate on test fold
2. Generate confusion matrix
3. Analyze per-class performance
4. Try full fine-tuning for 1-2% improvement
5. Experiment with data augmentation
6. Cross-validate on other folds

Good luck with training! 🚀

