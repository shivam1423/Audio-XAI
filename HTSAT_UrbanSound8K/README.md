# HTSAT UrbanSound8K Evaluation

This directory contains evaluation code for the **HTS-AT (Hierarchical Token Semantic Audio Transformer)** model on the **UrbanSound8K** dataset using an **AudioSet pre-trained checkpoint**.

## Overview

This folder is a separate evaluation setup for UrbanSound8K, keeping the original `HTSAT` folder untouched for XAI (explainable AI) workflows.

Based on the original HTSAT repository: [https://github.com/RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)

## Key Features

- **AudioSet Checkpoint Support**: Loads AudioSet pre-trained checkpoints (527 classes) and adapts them for UrbanSound8K (10 classes)
- **Transfer Learning**: Uses AudioSet features with a new classification head for UrbanSound8K
- **Complete Evaluation**: Provides accuracy, per-class metrics, and confusion matrices

## Dataset Structure

UrbanSound8K should be organized as:
```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   │   ├── 100032-0-0-0.wav
│   │   └── ...
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

The metadata CSV should have columns:
- `slice_file_name`: Audio filename
- `fold`: Fold number (1-10)
- `classID`: Integer class label (0-9)
- `class`: Class name (e.g., 'air_conditioner')

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the official HTSAT repository is available:
   - The code expects `HTS-Audio-Transformer` in `../HTSAT/HTS-Audio-Transformer/`
   - Or create a symlink if it's in a different location

## Usage

### Option 1: Using the Shell Script

Make the script executable and edit the paths:
```bash
chmod +x run_evaluation.sh
# Edit run_evaluation.sh to set CHECKPOINT, AUDIO_DIR, METADATA paths
./run_evaluation.sh
```

### Option 2: Using Python Directly

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

### Command-line Arguments

- `--checkpoint` (required): Path to AudioSet checkpoint file (.ckpt)
- `--audio_dir` (required): Path to UrbanSound8K audio directory (parent of fold1, fold2, etc.)
- `--metadata`: Path to UrbanSound8K.csv metadata file (optional, will auto-search)
- `--test_fold`: Test fold number (default: 10)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')
- `--output_dir`: Directory to save results (default: './results')
- `--num_classes`: Number of classes (default: 10 for UrbanSound8K)

## How It Works

1. **Model Loading**: 
   - Loads AudioSet checkpoint (527 classes)
   - Transfers all weights except the final classification head
   - Creates new classification head for 10 classes (UrbanSound8K)
   - Final layer is randomly initialized (transfer learning approach)

2. **Evaluation**:
   - Loads UrbanSound8K test data (default: fold 10)
   - Processes audio at 32kHz, 10-second clips
   - Generates predictions and computes metrics

## Output

The evaluation script generates the following outputs:

1. **predictions.csv** - Detailed predictions for each audio sample
   - Columns: filename, true_label, predicted_label, true_class, predicted_class, correct

2. **confusion_matrix.npy** - Confusion matrix (NumPy array)

3. **confusion_matrix.csv** - Confusion matrix in CSV format for easy viewing

4. **summary.txt** - Summary of evaluation results including:
   - Overall accuracy
   - Per-class accuracy
   - Configuration details

5. **Console output** - Includes:
   - Overall accuracy
   - Per-class precision, recall, F1-score
   - Classification report

## UrbanSound8K Classes

The dataset includes 10 urban sound classes:

0. **air_conditioner**
1. **car_horn**
2. **children_playing**
3. **dog_bark**
4. **drilling**
5. **engine_idling**
6. **gun_shot**
7. **jackhammer**
8. **siren**
9. **street_music**

## Expected Performance

Based on research, HTSAT AudioSet checkpoint achieves:
- **Zero-shot accuracy**: ~77% on UrbanSound8K
- This is competitive with other AudioSet pre-trained models

For better performance, consider fine-tuning the model on UrbanSound8K training folds.

## Model Architecture

HTSAT uses:
- Swin Transformer backbone with hierarchical design
- Mel spectrogram input (64 mel bins)
- Patch-based processing
- Multi-scale feature extraction
- AudioSet pre-training provides strong audio representations

## Notes

- **Important**: This is a transfer learning approach. The final classification layer is randomly initialized, so you're using AudioSet features but not AudioSet's classification directly.
- The model expects 32kHz audio (different from many other audio models that use 16kHz)
- Each audio clip is processed as a 10-second segment (shorter clips are repeated/padded)
- GPU recommended for faster evaluation
- The original HTSAT folder remains untouched for XAI workflows

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python evaluate_urbansound8k.py --batch_size 16 ...
```

### Missing Metadata File
The script will search for `UrbanSound8K.csv` in common locations. If not found, provide `--metadata` path explicitly.

### HTS-Audio-Transformer Not Found
Ensure the official HTSAT repository exists at:
- `../HTSAT/HTS-Audio-Transformer/` (relative to this folder)
- Or create a symlink: `ln -s /path/to/HTS-Audio-Transformer HTS-Audio-Transformer`

### Checkpoint Loading Issues
- Ensure checkpoint is from AudioSet (527 classes)
- Check PyTorch version compatibility
- Verify checkpoint file is not corrupted

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

## License

This code is provided for research purposes. Please refer to the original HTSAT repository for license information.

