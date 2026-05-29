# ResNet50 — Audio Classification with RISE Explainability

A complete pipeline for audio classification on the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset using a fine-tuned ResNet-50, with model-agnostic saliency explanations via RISE (Randomized Input Sampling for Explanations).

---

## Overview

This module:
1. Converts raw `.wav` audio to mel-spectrogram images on-the-fly
2. Fine-tunes a pretrained ResNet-50 using 5-fold cross-validation
3. Generates RISE saliency maps to highlight time-frequency regions driving predictions
4. Evaluates explanation quality using deletion and insertion causal metrics

---

## Directory Structure

```
ResNet50/
├── training/
│   ├── fine_tune_resnet.py       # Main training script (5-fold CV)
│   └── make_spectrograms.py      # (Legacy) Pre-compute spectrograms as PNGs
├── models/                       # Reserved for model checkpoints
├── explanations.py               # RISE and RISEBatch classes
├── evaluation.py                 # Deletion/insertion causal metrics
├── Saliency.py                   # Saliency map generation script
├── utils.py                      # Audio preprocessing, dataset, utilities
├── resnet50_esc50_per_file_inference.py  # Per-file inference on all ESC-50 clips
├── run.sh                        # SLURM job: training
├── run_saliency.sh               # SLURM job: saliency generation
├── run_evaluation.sh             # SLURM job: evaluation
├── requirements.txt
└── ESC50/                        # Expected dataset location
    ├── audio/                    # 2000 .wav files
    └── meta/esc50.csv
```

---

## Model

**Architecture**: ResNet-50 (pretrained on ImageNet, final FC layer replaced for 50 classes)
**Input**: 3-channel RGB image (224×224) — grayscale mel-spectrogram replicated across channels
**Output**: 50-class softmax probabilities

### Audio Preprocessing Pipeline

```
Raw Audio (.wav)
  → librosa.load at 22050 Hz
  → Mel-Spectrogram (128 bins, n_fft=1024, hop_length=512)
  → Convert to dB scale
  → Normalize to [0, 255]
  → Resize to 224×224 (BILINEAR)
  → Replicate to 3 channels
  → Normalize (mean=0.5, std=0.5 per channel)
  → ResNet-50 input (3×224×224)
```

Direct audio-to-spectrogram processing (no intermediate PNG files) avoids 8-bit quantization loss and ensures identical preprocessing between training and evaluation.

---

## Dataset: ESC-50

- **2000 labeled audio clips** across **50 environmental sound categories**
- 5 seconds each at 44.1 kHz
- Official 5-fold split encoded in `meta/esc50.csv`

Download from: https://github.com/karolpiczak/ESC-50

Place the dataset at `ResNet50/ESC50/` (or pass `--esc50_root` to override).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install the matching CUDA build of PyTorch separately (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### 2. Train (5-fold cross-validation)

```bash
python -m training.fine_tune_resnet \
    --esc50_root ESC50 \
    --epochs 50 \
    --batch_size 32 \
    --kfolds 5
```

Outputs: `resnet50_esc50_fold{1..5}.pt`

### 3. Generate saliency maps

```bash
python Saliency.py
```

Outputs: `saliency_outputs/fold_{1..5}/<filename>_saliency.png`

### 4. Per-file inference

```bash
python resnet50_esc50_per_file_inference.py \
    --esc50_root ESC50 \
    --checkpoints_dir . \
    --output_dir results/
```

Outputs: `esc50_predictions_resnet50_fold{1..5}.csv` (filename, true class, predicted class, confidence)

### 5. Causal metric evaluation

```bash
python evaluation.py
```

Outputs: `evaluation_graphs/fold_{1..5}/` — per-sample deletion/insertion curves and fold-level summary

---

## Explainability: RISE

**RISE** (Randomized Input Sampling for Explanations) is a model-agnostic technique that requires no gradient access.

**Algorithm**:
1. Sample N random binary masks (default: 6000, 8×8 grid upsampled to 224×224, coverage p1=0.1)
2. Apply each mask element-wise to the input spectrogram
3. Run model inference on all masked inputs
4. Saliency score per pixel = weighted average of prediction confidence across masks containing that pixel

The result is a 224×224 heatmap indicating which time-frequency regions most influenced the prediction.

Two implementations are available:
- `RISE` — single-sample
- `RISEBatch` — batch-optimized for multiple samples

---

## Evaluation: Deletion & Insertion Metrics

Causal metrics quantify whether highlighted regions are genuinely important:

| Metric | Process | Goal |
|--------|---------|------|
| **Deletion** | Progressively remove most-salient pixels (replaced with Gaussian blur) | Confidence should drop sharply |
| **Insertion** | Progressively reveal most-salient pixels (from zero baseline) | Confidence should rise sharply |

AUC of the resulting confidence curves summarizes explanation quality per sample.

---

## Cluster Execution (SLURM)

```bash
sbatch run.sh            # Training (2 GPUs, 20 GB RAM, 8-day timeout)
sbatch run_saliency.sh   # Saliency generation (1 GPU)
sbatch run_evaluation.sh # Evaluation (1 GPU)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0`, `torchvision >= 0.15` | Model training and inference |
| `librosa >= 0.10` | Audio loading and mel-spectrogram computation |
| `Pillow >= 10.0` | Spectrogram image handling |
| `scikit-image >= 0.19, < 0.22` | Mask resizing in RISE |
| `scikit-learn >= 1.3` | Cross-validation utilities |
| `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm` | General scientific computing |

See `requirements.txt` for pinned versions.
