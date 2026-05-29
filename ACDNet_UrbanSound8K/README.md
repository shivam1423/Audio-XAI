# ACDNet for UrbanSound8K Classification

Implementation of ACDNet (Adaptively Combined Dilated Network) for UrbanSound8K environmental sound classification.

Based on the paper: **"Environmental Sound Classification on the Edge: A Pipeline for Deep Acoustic Networks on Extremely Resource-Constrained Devices"** (arXiv:2103.03483)

## Overview

This implementation adapts ACDNet for the UrbanSound8K dataset with:
- Raw waveform input at 20kHz sampling rate
- 1.5-second audio clips
- BC (Between-Class) Learning for data augmentation
- 10-crop evaluation strategy
- Optimized for cluster/GPU training

## Project Structure

```
ACDNet_UrbanSound8K/
├── config/
│   └── config.py              # Configuration parameters
├── data/
│   ├── dataset.py             # UrbanSound8K dataset loader
│   └── preprocessor.py        # Audio preprocessing utilities
├── models/
│   └── acdnet.py             # ACDNet model architecture
├── training/
│   ├── train_generator.py    # BC Learning data generator
│   └── trainer.py            # Training loop implementation
├── evaluation/
│   └── evaluator.py          # Evaluation with 10-crop testing
├── utils/
│   └── helpers.py            # Utility functions
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Main evaluation script
│   └── run_train.sh          # SLURM submission script
├── trained_models/           # Saved model checkpoints
├── results/                  # Evaluation results
└── README.md
```

## Requirements

Install dependencies:

```bash
pip install torch>=1.10.0 torchaudio>=0.10.0 librosa>=0.9.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 matplotlib>=3.5.0 tqdm>=4.62.0
```

Or create a `requirements.txt`:

```
torch>=1.10.0
torchaudio>=0.10.0
librosa>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.62.0
```

Then install:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### UrbanSound8K Dataset Structure

The dataset should be organized with audio files in fold subdirectories:

```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   │   ├── 100032-3-0-0.wav
│   │   └── ...
│   ├── fold2/
│   │   └── ...
│   ├── ...
│   └── fold10/
│       └── ...
└── metadata/
    └── UrbanSound8K.csv
```

### Data Preprocessing (Required Before Training)

This implementation follows the original ACDNet methodology with **two-step preprocessing**:

#### Step 1: Prepare Raw Audio NPZ (Run Once)

Convert raw audio files to a single NPZ file for fast loading:

```bash
python scripts/prepare_urbansound8k.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./data \
    --sr 20000 \
    --verify
```

**What this does:**
- Loads all 8,732 audio files from UrbanSound8K
- Resamples to 20kHz (ACDNet requirement)
- Saves to `./data/urbansound8k_20k.npz` (~500MB)
- Takes 30-60 minutes (but only needed once!)

#### Step 2: Prepare Multi-Crop Validation Data (Run Once)

Generate pre-processed validation/test data with 10 evenly-spaced crops per sample:

```bash
python scripts/prepare_validation_data.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./val_data \
    --val_fold 9 \
    --test_fold 10
```

**What this does:**
- Creates 10 evenly-spaced crops per audio sample (following original ACDNet)
- Applies padding and normalization
- Saves to `./val_data/fold9_val10crop.npz` and `./val_data/fold10_val10crop.npz`
- Enables accurate validation with multi-crop averaging
- Takes ~5 minutes

**Why multi-crop preprocessing is critical:**
- Original ACDNet uses 10-crop averaging for validation/testing
- Single random crop: volatile accuracy (18-54%)
- Multi-crop averaging: stable accuracy (75-85%+)
- Matches paper methodology exactly

## Configuration

Key parameters are defined in `config/config.py` (updated to match original ACDNet):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 20 kHz | Audio sampling rate |
| Input Length | 30,225 samples | ~1.51 seconds (matching original ACDNet) |
| Batch Size | 64 | Training batch size (original ACDNet default) |
| Epochs | 500 | Total training epochs |
| Learning Rate | 0.1 | Initial learning rate |
| LR Schedule | [0.3, 0.6, 0.9] | Decay at epoch 150, 300, 450 |
| Warmup | 10 epochs | Learning rate warmup period |
| Optimizer | SGD + Nesterov | Momentum = 0.9 |
| Weight Decay | 5e-4 | L2 regularization |
| Train Folds | 1-8 | Folds used for training (7,079 samples) |
| Val Fold | 9 | Fold used for validation (816 samples) |
| Test Fold | 10 | Fold used for testing (837 samples) |
| N Crops | 10 | Multi-crop evaluation (10 evenly-spaced crops) |

## Usage

### Quick Start (Automated - Recommended)

The SLURM script automatically handles all preprocessing steps:

1. Edit `scripts/run_train.sh` and update the `DATA_DIR` variable:

```bash
DATA_DIR="../UrbanSound8K"  # Point to your UrbanSound8K dataset
```

2. Submit the job:

```bash
cd ACDNet_UrbanSound8K
sbatch scripts/run_train.sh
```

The script will automatically:
- Step 1: Prepare raw audio NPZ (if not exists)
- Step 1.5: Prepare multi-crop validation data (if not exists)
- Step 2: Train model with BC Learning
- Step 3: Evaluate on test set

3. Monitor progress:

```bash
tail -f slurm-*.out
```

### Manual Workflow (Step-by-Step)

#### Step 1: Prepare Raw Audio NPZ

```bash
python scripts/prepare_urbansound8k.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./data \
    --sr 20000 \
    --verify
```

Creates: `./data/urbansound8k_20k.npz` (~500MB)

#### Step 2: Prepare Multi-Crop Validation Data

```bash
python scripts/prepare_validation_data.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./val_data \
    --val_fold 9 \
    --test_fold 10 \
    --input_length 30225 \
    --n_crops 10
```

Creates: 
- `./val_data/fold9_val10crop.npz` (validation with 10 crops)
- `./val_data/fold10_val10crop.npz` (test with 10 crops)

#### Step 3: Train Model

```bash
python scripts/train.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./trained_models \
    --device cuda \
    --batch_size 64 \
    --epochs 500 \
    --lr 0.1 \
    --seed 42
```

#### Step 4: Evaluate Model

```bash
python scripts/evaluate.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --model_path ./trained_models/acdnet_us8k_best.pt \
    --output_dir ./results \
    --device cuda \
    --test_fold 10 \
    --n_crops 10
```

### Command Line Arguments

#### prepare_urbansound8k.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | required | Path to UrbanSound8K dataset |
| `--output_dir` | str | `./data` | Directory to save NPZ file |
| `--sr` | int | `20000` | Target sample rate |
| `--verify` | flag | False | Verify NPZ after creation |

#### prepare_validation_data.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--npz_path` | str | required | Path to urbansound8k_20k.npz |
| `--output_dir` | str | `./val_data` | Directory to save multi-crop data |
| `--val_fold` | int | `9` | Validation fold number |
| `--test_fold` | int | `10` | Test fold number |
| `--input_length` | int | `30225` | Input length for crops |
| `--n_crops` | int | `10` | Number of crops per sample |

#### train.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--npz_path` | str | required | Path to urbansound8k_20k.npz |
| `--output_dir` | str | `./trained_models` | Directory to save models |
| `--device` | str | `cuda` | Device (cuda or cpu) |
| `--batch_size` | int | `64` | Training batch size |
| `--epochs` | int | `500` | Number of epochs |
| `--lr` | float | `0.1` | Initial learning rate |
| `--seed` | int | `42` | Random seed |

#### evaluate.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--npz_path` | str | required | Path to urbansound8k_20k.npz |
| `--model_path` | str | required | Path to model checkpoint |
| `--output_dir` | str | `./results` | Directory to save results |
| `--device` | str | `cuda` | Device (cuda or cpu) |
| `--test_fold` | int | `10` | Test fold number |

## Model Architecture

ACDNet consists of two main components:

1. **SFEB (Spatial Feature Extraction Block)**: Convolutional filter bank that processes raw waveforms
2. **TFEB (Temporal Feature Extraction Block)**: Temporal feature processing with pooling and fully connected layers

### Model Specifications

- Input: Raw waveform (1, 1, 1, 30225)
- Parameters: ~4.74 million
- Model Size: ~18 MB
- Output: 10-class probabilities (UrbanSound8K classes)

**Note:** Input length is 30,225 samples (~1.51s at 20kHz) to match the original ACDNet implementation for ESC-50.

## BC (Between-Class) Learning

ACDNet uses BC Learning for data augmentation:

1. Randomly select two samples from different classes
2. Mix them with random ratio `r`: `mixed = r * sound1 + (1-r) * sound2`
3. Create soft label: `label = r * label1 + (1-r) * label2`
4. Train with KL Divergence loss

This technique improves generalization and robustness.

## Expected Results

Based on the original paper (arXiv:2103.03483):

| Model | Dataset | Accuracy | Model Size | Parameters |
|-------|---------|----------|------------|------------|
| ACDNet | UrbanSound8K | **84.45% ± 0.05%** | ~18 MB | ~4.74M |

**Important Notes:**
- The paper reports results using **10-fold cross-validation** (average of 10 separate training runs)
- This implementation uses a **single fold split** (train: folds 1-8, val: fold 9, test: fold 10)
- Single fold results may vary but should achieve **75-85%** accuracy on the test set
- Multi-crop validation (10 crops averaged) is essential for stable, accurate results

### Training Time

**Preprocessing (one-time):**
- Raw audio NPZ: ~30-60 minutes
- Multi-crop validation data: ~5 minutes

**Training:**
- Estimated: 15-20 hours on single GPU (GTX 1080 Ti)
- 500 epochs with batch size 64
- ~2-3 minutes per epoch

## Output Files

### After Training

- `trained_models/acdnet_us8k_best.pt`: Best model checkpoint
- `slurm_*.out`: Training logs (if using SLURM)

### After Evaluation

- `results/evaluation_results.json`: Complete evaluation metrics
- `results/predictions.npy`: Model predictions
- `results/labels.npy`: True labels
- `results/confusion_matrix.npy`: Confusion matrix

## Troubleshooting

### Common Issues

1. **NPZ file not found**
   ```
   FileNotFoundError: NPZ file not found: ./data/urbansound8k_20k.npz
   ```
   **Solution:** Run preprocessing first:
   ```bash
   python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data
   ```

2. **Multi-crop validation data not found**
   ```
   FileNotFoundError: Multi-crop validation data not found: ./val_data/fold9_val10crop.npz
   ```
   **Solution:** Run multi-crop preparation:
   ```bash
   python scripts/prepare_validation_data.py --npz_path ./data/urbansound8k_20k.npz --output_dir ./val_data
   ```

3. **Low validation accuracy (< 50%)**
   - This indicates validation is NOT using multi-crop averaging
   - Ensure `./val_data/fold9_val10crop.npz` exists and is loaded correctly
   - Check that validation uses pre-generated data (not on-the-fly random crops)

4. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 32` or `--batch_size 16`
   - Use CPU: `--device cpu` (much slower)

5. **Data Directory Not Found**
   - Verify path to UrbanSound8K dataset
   - Ensure structure: `UrbanSound8K/audio/fold1/`, `fold2/`, ..., `fold10/`
   - Check metadata exists: `UrbanSound8K/metadata/UrbanSound8K.csv`

6. **Shape mismatch errors**
   - Should be fixed with moveaxis transformation
   - Verify: `python scripts/verify_shapes.py`
   - Ensure NPZ preprocessing completed successfully

## UrbanSound8K Classes

The dataset contains 10 classes:

0. air_conditioner
1. car_horn
2. children_playing
3. dog_bark
4. drilling
5. engine_idling
6. gun_shot
7. jackhammer
8. siren
9. street_music

## Citation

If you use this implementation, please cite the original ACDNet paper:

```bibtex
@article{mohaimenuzzaman2022environmental,
  title={Environmental sound classification on the edge: A pipeline for deep acoustic networks on extremely resource-constrained devices},
  author={Mohaimenuzzaman, Md and Bergmeir, Christoph and West, Ian and Meyer, Bernd},
  journal={Pattern Recognition},
  pages={109025},
  year={2022},
  publisher={Elsevier}
}
```

## License

MIT License - See original ACDNet repository for details.

## Notes

- This implementation focuses on baseline ACDNet training and evaluation
- Pruning and quantization (Micro-ACDNet) are not included in this version
- The model is designed for raw waveform classification without spectrograms
- BC Learning significantly improves performance but increases training time
- 10-crop evaluation provides more robust accuracy estimates than single-crop

## Contact

For issues or questions about this implementation, please refer to the original ACDNet paper and repository.
