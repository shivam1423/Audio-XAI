# HTSAT Evaluation for ESC-50

This directory contains evaluation code for the **HTS-AT (Hierarchical Token Semantic Audio Transformer)** model on the ESC-50 dataset.

## Overview

Based on the original HTSAT repository: [https://github.com/RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)

## Configuration

- **Validation Fold**: 2
- **Training Folds**: 1, 3, 4, 5
- **Model Checkpoint**: `HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt`
- **Sample Rate**: 32kHz (HTSAT requirement)
- **Clip Length**: 10 seconds (320,000 samples at 32kHz)

## Files

- `htsat_config.py` - Configuration settings
- `htsat_model.py` - HTSAT model architecture (simplified version)
- `esc50_dataset.py` - ESC-50 dataset loader
- `evaluate.py` - Main evaluation script
- `run_evaluation.sh` - Bash script to run evaluation
- `requirements.txt` - Python dependencies

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Shell Script

Make the script executable and run:
```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

### Option 2: Using Python Directly

Run with default settings from config:
```bash
python evaluate.py
```

Run with custom settings:
```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio_dir /path/to/ESC50/audio \
    --val_fold 2 \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --output_dir ./results
```

### Command-line Arguments

- `--checkpoint`: Path to HTSAT checkpoint file (default: from config)
- `--audio_dir`: Path to ESC-50 audio directory (default: from config)
- `--metadata`: Path to esc50.csv metadata file (optional, will auto-detect)
- `--val_fold`: Validation fold number (default: 2)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')
- `--output_dir`: Directory to save results (default: './results')

## Output

The evaluation script generates the following outputs in the specified output directory:

1. **predictions.csv** - Detailed predictions for each audio sample
   - Columns: filename, true_label, predicted_label, correct

2. **confusion_matrix.npy** - Confusion matrix (NumPy array)

3. **confusion_matrix.png** - Visualization of confusion matrix

4. **evaluation_summary.txt** - Summary of evaluation results

5. **Console output** - Includes:
   - Overall accuracy
   - Per-class precision, recall, F1-score
   - Classification report

## Dataset Structure

The code expects ESC-50 audio files in the following format:
```
audio/
├── 1-100032-A-0.wav
├── 1-100210-A-1.wav
├── ...
└── 5-9032-A-0.wav
```

Filename format: `{fold}-{clip_id}-{take}-{target}.wav`

If an `esc50.csv` metadata file is available, it should be in one of:
- `../meta/esc50.csv` (relative to audio directory)
- `../esc50.csv`
- `audio/esc50.csv`

If not found, the script will create metadata from filenames.

## ESC-50 Classes

The dataset includes 50 environmental sound classes across 5 categories:

1. **Animals**: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
2. **Natural soundscapes**: rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm
3. **Human non-speech**: crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking/sipping
4. **Interior/domestic**: door wood knock, mouse click, keyboard typing, door wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking
5. **Exterior/urban**: helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw

## Model Architecture

HTSAT uses:
- Swin Transformer backbone with hierarchical design
- Mel spectrogram input (64 mel bins)
- Patch-based processing
- Multi-scale feature extraction

## Notes

- **Important**: This is a simplified implementation for evaluation purposes. For training and full features, please refer to the original repository.
- The model expects 32kHz audio (different from many other audio models that use 16kHz)
- Each audio clip is processed as a 10-second segment (original ESC-50 clips are 5 seconds, so they are repeated)
- GPU recommended for faster evaluation

## Citation

If you use HTSAT in your research, please cite:

```bibtex
@inproceedings{chen2022hts,
  title={HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  author={Chen, Ke and Du, Xingjian and Zhu, Bilei and Ma, Zejun and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={646--650},
  year={2022},
  organization={IEEE}
}
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python evaluate.py --batch_size 16
```

Or use CPU:
```bash
python evaluate.py --device cpu
```

### Missing Metadata File
The script will automatically create metadata from filenames if esc50.csv is not found.

### Checkpoint Loading Issues
The script attempts to load checkpoints flexibly. If issues persist, check:
- Checkpoint file path is correct
- Checkpoint was saved for ESC-50 (50 classes)
- PyTorch version compatibility

## License

This code is provided for research purposes. Please refer to the original HTSAT repository for license information.




