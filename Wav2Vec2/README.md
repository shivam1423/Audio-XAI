# Wav2Vec2 ESC-50 Fine-tuning

A ready-to-run PyTorch codebase for fine-tuning the original Wav2Vec2 model (`facebook/wav2vec2-base`) on the ESC-50 environmental sound classification dataset.

## What This Does

This project fine-tunes Wav2Vec2 (a pre-trained speech model) to classify 50 different environmental sounds like:
- Animal sounds (dog, cat, cow, etc.)
- Human sounds (laughing, coughing, sneezing, etc.) 
- Nature sounds (rain, thunderstorm, sea waves, etc.)
- Machine sounds (car horn, airplane, train, etc.)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download ESC-50 Dataset
```bash
python train.py --download_data
```

### 3. Train the Model
```bash
python train.py --batch_size 8 --num_epochs 10
```

### 4. Test on New Audio
```bash
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_file path/to/your/audio.wav
```

## Project Structure

```
Wav2Vec2/
├── config.py                 # Configuration settings
├── train.py                  # Main training script
├── inference.py              # Inference script
├── requirements.txt          # Python dependencies
├── data/
│   ├── __init__.py
│   └── dataset.py           # ESC-50 dataset loader
├── model/
│   ├── __init__.py
│   └── wav2vec2_classifier.py  # Wav2Vec2 model architecture
├── training/
│   ├── __init__.py
│   └── trainer.py           # Training loop and utilities
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
├── outputs/                 # Training outputs (plots, results)
├── checkpoints/             # Model checkpoints
└── logs/                    # TensorBoard logs
```

## How It Works

1. **Data Loading**: Loads ESC-50 audio files and preprocesses them (resampling, padding)
2. **Model Architecture**: Uses pre-trained Wav2Vec2 with a custom classification head
3. **Training**: Fine-tunes the model with different learning rates for pre-trained vs new layers
4. **Evaluation**: Tests on held-out data and generates confusion matrices

## Key Features

- **Modular Design**: Clean separation of data, model, and training code
- **Automatic Data Download**: Downloads ESC-50 dataset automatically
- **Flexible Training**: Command-line arguments for all hyperparameters
- **Visualization**: Generates training plots and confusion matrices
- **Checkpointing**: Saves best models and allows resuming training
- **Inference**: Easy-to-use script for testing on new audio files

## Training Options

```bash
# Basic training
python train.py

# Custom hyperparameters
python train.py --batch_size 16 --learning_rate 1e-4 --num_epochs 20

# Freeze Wav2Vec2 feature extractor (faster training)
python train.py --freeze_feature_extractor

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_5.pt

# Test only (requires checkpoint)
python train.py --test_only --resume checkpoints/best_model_wav2vec2.pt
```

## Inference Options

```bash
# Single file
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_file test.wav

# Directory of files
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_dir test_audio/

# Save results to CSV
python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_dir test_audio/ --output_file results.csv
```

## Expected Performance

With default settings, you should expect:
- **Training time**: ~2-4 hours on GPU (depending on hardware)
- **Test accuracy**: ~85-90% on ESC-50
- **Model size**: ~95M parameters (mostly from Wav2Vec2)

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU training)
- ~2GB disk space for ESC-50 dataset
- ~4GB RAM for training

## Troubleshooting

**Out of memory?** Reduce batch size:
```bash
python train.py --batch_size 4
```

**Training too slow?** Freeze feature extractor:
```bash
python train.py --freeze_feature_extractor
```

**Audio format issues?** The code supports WAV, MP3, FLAC, M4A, and OGG files.

## What You Get

After training, you'll have:
- Trained model checkpoints in `checkpoints/`
- Training plots in `outputs/`
- Confusion matrix visualization
- Detailed test results in JSON format
- TensorBoard logs for monitoring training

The model can then classify any 5-second audio clip into one of 50 environmental sound categories with high accuracy.

