# HTSAT UrbanSound8K Evaluation Guide

## Two Types of Evaluation

### 1. Zero-shot Evaluation (AudioSet checkpoint)
Uses `evaluate_urbansound8k.py` to evaluate AudioSet checkpoint directly.
- **Checkpoint**: `.ckpt` file (AudioSet pretrained)
- **Expected accuracy**: ~7-77% (depends on approach)
- **Use case**: Testing AudioSet features without training

### 2. Fine-tuned Model Evaluation (Trained checkpoint)
Uses `evaluate_trained_model.py` to evaluate your trained model.
- **Checkpoint**: `.pth` file (from training)
- **Expected accuracy**: 85-95%
- **Use case**: Testing your fine-tuned model

## Quick Start: Evaluate Trained Model

### Option 1: Shell Script (Recommended)

```bash
# Edit run_evaluate_trained.sh to set paths
./run_evaluate_trained.sh
```

### Option 2: Python Command

```bash
python evaluate_trained_model.py \
    --checkpoint training_output_fold10/best_model.pth \
    --audio_dir ../UrbanSound8K/audio \
    --metadata ../UrbanSound8K/metadata/UrbanSound8K.csv \
    --test_fold 10 \
    --batch_size 32 \
    --device cuda \
    --output_dir ./results_trained_fold10
```

## Understanding Checkpoints

### AudioSet Checkpoint (.ckpt)
```
HTSAT_AudioSet_Saved_1.ckpt
├── sed_model.layer1.weight
├── sed_model.layer2.weight
└── ... (527 classes)
```
- Pre-trained on AudioSet
- 527 output classes
- Use: `evaluate_urbansound8k.py`

### Trained Checkpoint (.pth)
```
training_output_fold10/best_model.pth
├── epoch: 8
├── model_state_dict: {...}  # Trained weights
├── optimizer_state_dict: {...}
├── val_acc: 0.8817
└── val_loss: 0.3456
```
- Fine-tuned on UrbanSound8K
- 10 output classes
- Use: `evaluate_trained_model.py` ✓

## Common Mistakes

### ❌ Wrong: Using `evaluate_urbansound8k.py` for trained models
```bash
# This will load trained .pth as AudioSet .ckpt and fail
python evaluate_urbansound8k.py --checkpoint training_output_fold10/best_model.pth
# Result: ~6% accuracy (random predictions)
```

### ✅ Correct: Using `evaluate_trained_model.py` for trained models
```bash
# This correctly loads trained .pth checkpoint
python evaluate_trained_model.py --checkpoint training_output_fold10/best_model.pth
# Result: ~88% accuracy (actual trained performance)
```

## Evaluation Options

### Basic Configuration

- `--checkpoint`: Path to trained checkpoint (required, `.pth` file)
- `--audio_dir`: Path to UrbanSound8K audio directory (required)
- `--metadata`: Path to UrbanSound8K.csv (auto-searched if not provided)
- `--test_fold`: Test fold number (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--device`: Device to use (default: cuda)
- `--output_dir`: Output directory (default: ./results_trained)

## Output Files

Evaluation saves to `--output_dir`:

```
results_trained_fold10/
├── predictions.csv              # Per-sample predictions
├── confusion_matrix.npy         # Confusion matrix (NumPy)
├── confusion_matrix.csv         # Confusion matrix (readable)
└── summary.txt                  # Evaluation summary
```

## Interpreting Results

### predictions.csv
```csv
filename,true_label,predicted_label,true_class,predicted_class,correct
7061-6-0-0.wav,0,0,air_conditioner,air_conditioner,True
```

### confusion_matrix.csv
Shows which classes are confused with each other.

### summary.txt
```
Test Accuracy: 88.17%

Per-Class Accuracy:
  air_conditioner: 75.00%
  car_horn: 90.91%
  children_playing: 99.00%
  ...
```

## Comparison: Zero-shot vs Fine-tuned

| Method | Script | Checkpoint | Accuracy | Use Case |
|--------|--------|------------|----------|----------|
| Zero-shot (random head) | `evaluate_urbansound8k.py` | `.ckpt` | 7.89% | Baseline |
| Zero-shot (semantic) | Custom mapping | `.ckpt` | ~77% | AudioSet features |
| Fine-tuned (freeze) | `evaluate_trained_model.py` | `.pth` | 85-90% | Fast training |
| Fine-tuned (full) | `evaluate_trained_model.py` | `.pth` | 88-95% | Best performance |

## Evaluating Different Checkpoints

### Best model (highest validation accuracy)
```bash
python evaluate_trained_model.py \
    --checkpoint training_output_fold10/best_model.pth \
    --test_fold 10
```

### Specific epoch
```bash
python evaluate_trained_model.py \
    --checkpoint training_output_fold10/checkpoint_epoch_15.pth \
    --test_fold 10
```

### Different fold (cross-validation)
```bash
python evaluate_trained_model.py \
    --checkpoint training_output_fold1/best_model.pth \
    --test_fold 1
```

## Troubleshooting

### Error: "model_state_dict not found"
- **Cause**: Using wrong evaluation script
- **Fix**: Use `evaluate_trained_model.py` for `.pth` files

### Low accuracy (~6-8%)
- **Cause**: Loading trained checkpoint with wrong script
- **Fix**: Use `evaluate_trained_model.py` instead of `evaluate_urbansound8k.py`

### CUDA out of memory
- **Fix**: Reduce batch size: `--batch_size 16` or `--batch_size 8`

### Different accuracy than training
- Training shows validation accuracy on fold used during training
- Evaluation shows test accuracy on held-out fold
- Small differences (±2%) are normal

## Verifying Your Results

Your training output showed:
```
Best validation accuracy: 88.17% (epoch 8)
Final Validation Accuracy: 88.17%
```

When you run `evaluate_trained_model.py` with the best checkpoint, you should see:
```
Overall Accuracy: ~88.17%
```

If you see much lower (~6%), you're using the wrong evaluation script!

## Next Steps

After successful evaluation:
1. ✅ Verify accuracy matches training results (~88%)
2. 📊 Analyze confusion matrix for error patterns
3. 🔍 Review per-class performance
4. 🎯 Identify weak classes (e.g., siren had lower recall)
5. 🚀 Try full fine-tuning for 1-2% improvement
6. 🔄 Cross-validate on other folds

Good luck with evaluation! 🎉

