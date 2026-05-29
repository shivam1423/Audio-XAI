# ResNet50 for UrbanSound8K Classification

Implementation of ResNet50 deep learning model for environmental sound classification using the UrbanSound8K dataset. This project uses mel spectrograms as input features and achieves competitive performance on the benchmark dataset.

## Features

- **ResNet50 Architecture**: Adapted from ImageNet-pretrained ResNet50 for audio classification
- **Mel Spectrogram Input**: Converts audio to mel spectrograms (128x128) for robust feature representation
- **Fold-Based Training**: Uses standard UrbanSound8K fold splits for reproducible results
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **GPU Support**: Optimized for CUDA-enabled GPUs with automatic CPU fallback
- **SLURM Integration**: Ready-to-use scripts for cluster computing environments

## Project Structure

```
Resnet50_UrbanSound8K/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── dataset.py             # UrbanSound8K dataset loader
│   └── preprocessor.py        # Audio preprocessing (mel spectrograms)
├── models/
│   ├── __init__.py
│   └── resnet50.py            # ResNet50 architecture (10 classes)
├── training/
│   ├── __init__.py
│   └── trainer.py             # Training loop with logging
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py           # Evaluation metrics and testing
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── run_train.sh           # SLURM submission script
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Utility functions
├── trained_models/            # Saved model checkpoints
├── results/                   # Evaluation results
├── requirements.txt
└── README.md
```

## Requirements

### Software Dependencies

- Python 3.7+
- PyTorch 1.10.0+
- CUDA 11.0+ (optional, for GPU acceleration)

### Python Packages

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch>=1.10.0 torchvision>=0.11.0 librosa>=0.9.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 tqdm>=4.62.0 soundfile>=0.11.0 matplotlib>=3.5.0 seaborn>=0.11.0
```

## Dataset Setup

### UrbanSound8K Dataset

Download the UrbanSound8K dataset from the [official website](https://urbansounddataset.weebly.com/urbansound8k.html) or academic resources.

### Expected Directory Structure

The dataset should be organized as follows:

**Option 1: Standard Structure**
```
UrbanSound8K/
├── fold1/
│   ├── 100032-3-0-0.wav
│   ├── 100263-2-0-117.wav
│   └── ...
├── fold2/
│   └── ...
├── ...
├── fold10/
│   └── ...
└── UrbanSound8K.csv
```

**Option 2: With Audio Subdirectory**
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

The loader automatically detects and handles both structures.

### Dataset Information

- **Classes**: 10 urban sound categories
- **Total Samples**: ~8,700 audio clips
- **Folds**: 10 pre-defined folds for cross-validation
- **Audio Format**: WAV files, variable length (typically 1-4 seconds)
- **Sample Rates**: Variable (typically 44.1 kHz or 48 kHz)

### Class Labels

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

## Configuration

Key parameters are defined in `config/config.py` and can be overridden via command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr` | 22050 | Audio sampling rate (Hz) |
| `duration` | 4.0 | Audio duration (seconds) |
| `n_mels` | 128 | Number of mel bands |
| `spec_height` | 128 | Spectrogram height |
| `spec_width` | 128 | Spectrogram width |
| `batch_size` | 32 | Training batch size |
| `n_epochs` | 100 | Number of training epochs |
| `lr` | 0.001 | Initial learning rate |
| `optimizer_type` | sgd | Optimizer (sgd or adam) |
| `train_folds` | [1-8] | Training folds |
| `val_fold` | 9 | Validation fold |
| `test_fold` | 10 | Test fold |

## Usage

### Training

#### Option 1: Using SLURM (Recommended for Clusters)

1. Edit `scripts/run_train.sh` and update the dataset path:

```bash
DATA_DIR="../UrbanSound8K"
```

2. Submit the SLURM job:

```bash
cd Resnet50_UrbanSound8K
sbatch scripts/run_train.sh
```

3. Monitor progress:

```bash
tail -f slurm-*.out
```

#### Option 2: Direct Python Execution

**Basic Training:**

```bash
python scripts/train.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./trained_models \
    --device cuda \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --seed 42
```

**Training with Custom Parameters:**

```bash
python scripts/train.py \
    --data_dir /path/to/UrbanSound8K \
    --output_dir ./my_models \
    --device cuda \
    --batch_size 64 \
    --epochs 150 \
    --lr 0.01 \
    --optimizer adam \
    --num_workers 8 \
    --seed 42
```

**Training on CPU:**

```bash
python scripts/train.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./trained_models \
    --device cpu \
    --batch_size 16
```

### Command-Line Arguments (train.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | required | Path to UrbanSound8K dataset |
| `--output_dir` | str | ./trained_models | Directory to save models |
| `--device` | str | cuda | Device (cuda or cpu) |
| `--batch_size` | int | 32 | Training batch size |
| `--epochs` | int | 100 | Number of epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--optimizer` | str | sgd | Optimizer (sgd or adam) |
| `--num_workers` | int | 4 | Data loading workers |
| `--seed` | int | 42 | Random seed |
| `--no_pretrained` | flag | False | Don't use pretrained weights |

### Evaluation

After training, evaluate the model on the test set:

```bash
python scripts/evaluate.py \
    --data_dir ../UrbanSound8K \
    --model_path ./trained_models/best_model.pth \
    --output_dir ./results \
    --device cuda
```

### Command-Line Arguments (evaluate.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | required | Path to UrbanSound8K dataset |
| `--model_path` | str | required | Path to model checkpoint |
| `--output_dir` | str | ./results | Directory to save results |
| `--device` | str | cuda | Device (cuda or cpu) |
| `--batch_size` | int | 32 | Evaluation batch size |
| `--test_fold` | int | 10 | Test fold number |
| `--num_workers` | int | 4 | Data loading workers |
| `--seed` | int | 42 | Random seed |

## Model Architecture

### ResNet50 Adaptations

1. **Input Layer Modification**:
   - Original: `Conv2d(3, 64, ...)` for RGB images
   - Modified: `Conv2d(1, 64, ...)` for single-channel spectrograms
   - Pretrained weights averaged across channels

2. **Output Layer Modification**:
   - Original: `Linear(2048, 1000)` for ImageNet
   - Modified: `Linear(2048, 10)` for UrbanSound8K

3. **Architecture Flow**:
   ```
   Input (1x128x128) → Conv1 → BatchNorm → ReLU → MaxPool
   → ResNet Block 1 (64 channels)
   → ResNet Block 2 (128 channels)
   → ResNet Block 3 (256 channels)
   → ResNet Block 4 (512 channels)
   → Global Average Pooling (2048 features)
   → Fully Connected (10 classes)
   ```

### Model Statistics

- **Total Parameters**: ~25.5 million
- **Trainable Parameters**: ~25.5 million
- **Model Size**: ~97 MB
- **Input Shape**: (batch, 1, 128, 128)
- **Output Shape**: (batch, 10)

## Training Details

### Data Preprocessing Pipeline

1. **Audio Loading**: Load WAV file, resample to 22050 Hz
2. **Duration Normalization**: Pad or truncate to 4 seconds (88,200 samples)
3. **Mel Spectrogram**: Compute mel spectrogram (128 mel bands, fmax=8000 Hz)
4. **Power to dB**: Convert power spectrogram to dB scale
5. **Normalization**: Normalize to [0, 1] range
6. **Resize**: Fix time dimension to 128 frames

### Training Configuration

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: SGD with momentum (0.9) and weight decay (1e-4)
- **Learning Rate Schedule**: ReduceLROnPlateau (patience=10, factor=0.1)
- **Batch Size**: 32
- **Epochs**: 100
- **Early Stopping**: Based on validation loss

### Data Splits

- **Training**: Folds 1-8 (~7,000 samples)
- **Validation**: Fold 9 (~800 samples)
- **Test**: Fold 10 (~800 samples)

## Output Files

### After Training

The following files are saved in the `trained_models/` directory:

- `best_model.pth`: Best model checkpoint (highest validation accuracy)
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every 10 epochs)
- `training_log_YYYYMMDD_HHMMSS.csv`: Training history (loss, accuracy, LR per epoch)
- `config.json`: Configuration used for training

### After Evaluation

The following files are saved in the `results/` directory:

- `evaluation_results.json`: Complete evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `per_class_metrics.png`: Per-class performance bar chart
- `confusion_matrix.npy`: Raw confusion matrix
- `predictions.npy`: Model predictions
- `labels.npy`: True labels

## Expected Performance

Based on literature and experiments with ResNet-based models on UrbanSound8K:

| Metric | Expected Range |
|--------|----------------|
| **Test Accuracy** | 78-84% |
| **Training Time** | 4-6 hours (single GPU) |
| **Convergence** | 50-70 epochs |
| **Best Val Loss** | 0.3-0.5 |

### Per-Class Performance

Some classes are inherently easier to classify than others:
- **High accuracy**: gun_shot, dog_bark, siren
- **Moderate accuracy**: children_playing, street_music
- **Challenging**: air_conditioner, engine_idling (similar acoustic properties)

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
python scripts/train.py --data_dir ../UrbanSound8K --batch_size 16

# Or use CPU
python scripts/train.py --data_dir ../UrbanSound8K --device cpu
```

**2. Dataset Not Found**

```bash
# Verify dataset structure
ls ../UrbanSound8K/
# Should see: fold1/, fold2/, ..., fold10/, UrbanSound8K.csv

# Check metadata file location
find ../UrbanSound8K -name "UrbanSound8K.csv"
```

**3. Module Import Errors**

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version (3.7+ required)
python --version

# Run from project root
cd Resnet50_UrbanSound8K
python scripts/train.py --data_dir ../UrbanSound8K
```

**4. Slow Training**

```bash
# Increase number of data loading workers
python scripts/train.py --data_dir ../UrbanSound8K --num_workers 8

# Verify GPU usage
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**5. LibriSA/SoundFile Errors**

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libsndfile1

# Install system dependencies (macOS)
brew install libsndfile
```

## Advanced Usage

### Modifying Configuration

Edit `config/config.py` to change default parameters:

```python
# Example: Increase mel bands for finer frequency resolution
n_mels = 256  # instead of 128

# Example: Change learning rate schedule
scheduler_type = 'cosine'  # instead of 'plateau'
```

### Custom Data Augmentation

Extend the `AudioPreprocessor` class in `data/preprocessor.py`:

```python
def add_noise(self, audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise
```

### Transfer Learning

Use the trained model as a feature extractor for other audio tasks:

```python
from models.resnet50 import ResNet50AudioClassifier

model = ResNet50AudioClassifier(num_classes=10)
model.load_state_dict(torch.load('trained_models/best_model.pth'))

# Extract features
features = model.get_features(audio_input)
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{salamon2014dataset,
  title={A dataset and taxonomy for urban sound research},
  author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle={Proceedings of the 22nd ACM international conference on Multimedia},
  pages={1041--1044},
  year={2014}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- UrbanSound8K dataset by J. Salamon, C. Jacoby, and J. P. Bello
- ResNet architecture by K. He, X. Zhang, S. Ren, and J. Sun
- PyTorch framework by Facebook AI Research

## Contact

For questions, issues, or contributions, please refer to the project repository or contact the maintainers.

---

**Note**: This implementation is designed for research and educational purposes. For production deployments, consider additional optimizations such as model quantization, pruning, or distillation.
