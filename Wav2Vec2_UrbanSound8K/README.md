# Wav2Vec2 UrbanSound8K Fine-tuning

A ready-to-run PyTorch codebase for fine-tuning the Wav2Vec2 model (`facebook/wav2vec2-base`) on the UrbanSound8K dataset for urban sound classification.

## Overview

This project fine-tunes Wav2Vec2 (a pre-trained speech model) to classify 10 different urban sound categories:
- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- engine_idling
- gun_shot
- jackhammer
- siren
- street_music

## Dataset Structure

The code expects the UrbanSound8K dataset at `../UrbanSound8K/` (relative path for cluster compatibility) with the following structure:

```
../UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   ├── fold4/
│   ├── fold5/
│   ├── fold6/
│   ├── fold7/
│   ├── fold8/
│   ├── fold9/
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

### Data Split

By default, the training uses:
- **Training**: Folds 1-8 (~7,000 samples)
- **Validation**: Fold 9 (~870 samples)
- **Test**: Fold 10 (~870 samples)

This can be customized via command-line arguments.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Local)

```bash
python train.py --data_dir ../UrbanSound8K --batch_size 16 --num_epochs 30
```

### 3. Train on Cluster

Submit the SLURM job:

```bash
sbatch run_train.sh
```

Edit `run_train.sh` to adjust:
- GPU requirements (`#SBATCH --gres=gpu:1`)
- Time limit (`#SBATCH --time=24:00:00`)
- Memory (`#SBATCH --mem=32G`)
- Module loading commands for your cluster

### 4. Evaluate Trained Model

```bash
# Local
python train.py --data_dir ../UrbanSound8K --test_only --resume checkpoints/best_model_wav2vec2.pt

# Cluster
sbatch run_eval.sh
```

### 5. Run Inference on New Audio

```bash
# Single file
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_file path/to/audio.wav

# Directory of files
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_dir path/to/audio_dir/

# Save results to CSV
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_dir path/to/audio_dir/ --output_file results.csv
```

## Project Structure

```
Wav2Vec2_UrbanSound8K/
├── config.py                      # Configuration (10 classes, relative paths)
├── train.py                       # Main training script
├── inference.py                   # Inference script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore patterns
├── run_train.sh                   # Cluster training script (SLURM)
├── run_eval.sh                    # Cluster evaluation script (SLURM)
├── data/
│   ├── __init__.py
│   └── urbansound8k_dataset.py   # Dataset loader with fold support
├── model/
│   ├── __init__.py
│   └── wav2vec2_classifier.py    # Wav2Vec2 model architecture
├── training/
│   ├── __init__.py
│   └── trainer.py                # Training loop and utilities
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Utility functions
├── checkpoints/                   # Saved model checkpoints
├── outputs/                       # Training plots and results
└── logs/                          # TensorBoard logs
```

## Training Options

```bash
# Basic training
python train.py --data_dir ../UrbanSound8K

# Custom hyperparameters
python train.py \
    --data_dir ../UrbanSound8K \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --num_epochs 30 \
    --weight_decay 0.01

# Freeze Wav2Vec2 feature extractor (faster training, may reduce accuracy)
python train.py --freeze_feature_extractor

# Custom fold split
python train.py \
    --train_folds 1 2 3 4 5 6 7 8 9 \
    --val_fold 10 \
    --test_fold 10

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pt

# Test only (requires checkpoint)
python train.py --test_only --resume checkpoints/best_model_wav2vec2.pt
```

## Configuration

Key settings in `config.py`:

- `NUM_CLASSES = 10` - UrbanSound8K has 10 classes
- `SAMPLE_RATE = 16000` - Wav2Vec2 standard sample rate
- `MAX_DURATION = 4.0` - Audio clips padded/trimmed to 4 seconds
- `DATA_DIR = "../UrbanSound8K"` - Relative path for cluster
- `BATCH_SIZE = 16` - Default batch size
- `LEARNING_RATE = 3e-4` - Default learning rate
- `NUM_EPOCHS = 30` - Default number of epochs

## Expected Performance

### Training Time
- **Single GPU (V100/A100)**: ~2-3 hours for 30 epochs
- **CPU**: Not recommended (extremely slow)

### Model Performance
- **Expected Test Accuracy**: 75-85% on fold 10
- **Model Size**: ~95M parameters (mostly from pre-trained Wav2Vec2)
- **Inference Speed**: ~20-30 samples/second on GPU

### Comparison with Other Models
For reference, state-of-the-art models on UrbanSound8K:
- CNN-based models: 70-80%
- HTSAT (Audio Transformer): 80-90%
- Wav2Vec2 (this implementation): 75-85%

## Output Files

After training, you'll find:

1. **Checkpoints** (`checkpoints/`)
   - `best_model_wav2vec2.pt` - Best model based on validation accuracy
   - `checkpoint_epoch_*.pt` - Regular checkpoints (every 5 epochs)

2. **Visualizations** (`outputs/`)
   - `training_history.png` - Loss and accuracy curves
   - `confusion_matrix.png` - Confusion matrix on test set
   - `test_results.json` - Detailed test metrics

3. **Logs** (`logs/`)
   - TensorBoard logs for monitoring training

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open `http://localhost:6006` in your browser.

## Troubleshooting

### Out of Memory?
Reduce batch size:
```bash
python train.py --batch_size 8
```

### Training Too Slow?
Freeze feature extractor:
```bash
python train.py --freeze_feature_extractor
```

### Dataset Not Found?
Ensure the dataset path is correct:
```bash
ls ../UrbanSound8K/audio/fold1/
ls ../UrbanSound8K/metadata/UrbanSound8K.csv
```

### Audio Format Issues?
The code supports: WAV, MP3, FLAC, M4A, OGG formats.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU training)
- ~2GB disk space for model and outputs
- ~4GB RAM minimum (16GB+ recommended for training)

## Integration with RISE Framework

This model can be used with the RISE explainability framework:
- Checkpoint format is compatible with RISE evaluation scripts
- Model architecture supports saliency map generation
- Can be compared with HTSAT results on the same dataset

## Citation

If you use this code, please cite:

**Wav2Vec2:**
```
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  journal={arXiv preprint arXiv:2006.11477},
  year={2020}
}
```

**UrbanSound8K:**
```
@inproceedings{salamon2014dataset,
  title={A dataset and taxonomy for urban sound research},
  author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle={22nd ACM international conference on Multimedia},
  pages={1041--1044},
  year={2014}
}
```

## License

This project is provided as-is for research purposes. Please respect the licenses of:
- Wav2Vec2 model (Facebook AI)
- UrbanSound8K dataset
- PyTorch and other dependencies

## Contact

For issues or questions, please open an issue in the repository.
