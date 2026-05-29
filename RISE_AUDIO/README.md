# RISE-AUDIO

RISE (Randomized Input Sampling for Explanation) adapted for audio classification. The method generates saliency maps showing which parts of a sound — in time and frequency — drove a model's prediction. Works with four different models and two datasets without touching model internals.

---

## How it works

The core idea is borrowed from image RISE: randomly mask parts of the input, run the masked version through the model, and correlate which masks led to higher/lower predictions. Do this 6000 times and you get a saliency map.

For audio, masking happens in the time-frequency domain. An audio file is converted to a spectrogram via STFT, masks are applied there, and then reconstructed back to audio via inverse STFT before being fed into the model. This way the framework stays model-agnostic — it doesn't need to know anything about model internals.

Four mask types are generated per run, each capturing a different audio structure:
- **Time stripes** — vertical bands that cover consecutive time frames
- **Frequency bands** — horizontal bands that cover frequency ranges
- **TF patches** — rectangular regions covering both dimensions
- **Mel-band masks** — structured by mel-frequency spacing (closer to how humans hear)

After saliency maps are generated, insertion and deletion AUC metrics are computed to measure explanation quality.

---

## Models

| Model | Input type | Sample rate | Checkpoint |
|---|---|---|---|
| ResNet50 | Mel spectrogram (224×224 RGB) | 22050 Hz | `checkpoints/resnet50_esc50.pt` |
| Wav2Vec2 | Raw waveform | 16000 Hz | `checkpoints/best_model_wav2vec2.pt` |
| HTSAT | Raw waveform (10s clips) | 32000 Hz | `checkpoints/HTSAT.ckpt` |
| ACDNet | Raw waveform (~1.5s clips) | 20000 Hz | `checkpoints/acdnet_weight_pruned_trained_fold4_90.50.pt` |

Internally, all masking runs at 48 kHz, then audio is resampled to each model's native sample rate before inference.

---

## Datasets

**ESC-50** — 50 environmental sound classes
Expected at: `../ESC50/audio/`

**ESC-10** — 10-class subset of ESC-50, used with Wav2Vec2
Expected at: `../ESC50/audio_esc10/`

**UrbanSound8K** — 10 urban sound classes, fold 10 used for evaluation
Expected at: `../UrbanSound8K/audio/fold10/`

Checkpoint paths for UrbanSound8K models live in `src/utils.py` under `DATASET_CONFIG`.

---

## Running the code

All scripts are designed for SLURM but can be run locally by replacing `srun python -u main.py` with `python -u main.py` and removing the `#SBATCH` lines.

Each script takes up to 5 positional arguments, all optional — defaults are set inside the script.

```bash
bash run_resnet.sh [DATASET] [INPUT_DIR] [OUTPUT_DIR] [SOFT_MASKING] [OCCLUSION]
```

### Scripts and their defaults

| Script | Model | Dataset | Default occlusion |
|---|---|---|---|
| `run_resnet.sh` | ResNet50 | ESC-50 | `black` |
| `run_resnet_us8k.sh` | ResNet50 | UrbanSound8K | `black` |
| `run_wav2vec2.sh` | Wav2Vec2 | ESC-10 | `freq` |
| `run_wav2vec2_us8k.sh` | Wav2Vec2 | UrbanSound8K | `freq` |
| `run_htsat.sh` | HTSAT | ESC-50 | `time` |
| `run_htsat_us8k.sh` | HTSAT | UrbanSound8K | `time` |
| `run_acdnet.sh` | ACDNet | ESC-50 | `black` |
| `run_acdnet_us8k.sh` | ACDNet | UrbanSound8K | `black` |

### Examples

Run ResNet on ESC-50 with all defaults:
```bash
bash run_resnet.sh
```

Run ResNet on UrbanSound8K:
```bash
bash run_resnet.sh urbansound8k
```

Override input and output directories:
```bash
bash run_resnet.sh esc50 /path/to/audio /path/to/output
```

Run HTSAT with Gaussian soft masking:
```bash
bash run_htsat.sh esc50 ../ESC50/audio results/saliency/htsat/test gaussian time
```

### Running without SLURM

Set the environment variables manually and call `main.py` directly:

```bash
export DATASET=esc50
export MODEL_TYPE=resnet
export INPUT_DIR=../ESC50/audio
export OUTPUT_DIR=results/saliency/resnet/test_run
export SOFT_MASKING=none
export OCCLUSION=black

python -u main.py
```

---

## What you can change

### Occlusion type (arg 5 in scripts)
Controls how masked regions are replaced:

| Value | What it does |
|---|---|
| `black` | Replace with zeros (silence) |
| `time` | Zero only along time axis |
| `freq` | Zero only along frequency axis |
| `gaussian_black` / `gaussian_time` / `gaussian_freq` | Gaussian noise variants |
| `discrete_black` / `discrete_time` / `discrete_freq` | Discrete replacement |

### Soft masking (arg 4 in scripts)
Controls mask edge smoothing:
- `none` — hard binary edges (default for all scripts)
- `gaussian` — Gaussian blur on mask edges
- `bilinear` — bilinear interpolation (original image RISE approach)

### Mask generation parameters
These live in `src/utils.py` and change the nature of the masks themselves:

```python
DEFAULT_N_MASKS = 6000          # More masks = smoother saliency, much slower
DEFAULT_TIME_STRIPE_FRAC = 0.25 # Proportion of masks that are time stripes
DEFAULT_FREQ_BAND_FRAC   = 0.25 # Proportion that are frequency bands
DEFAULT_RECT_PATCH_FRAC  = 0.25 # Proportion that are rectangular patches
DEFAULT_MEL_BAND_FRAC    = 0.25 # Proportion that are mel-spaced bands

TIME_STRIPE_WIDTH_PX  = (4, 24)  # Min/max stripe width in pixels
FREQ_BAND_HEIGHT_PX   = (4, 24)  # Min/max band height in pixels
RECT_SIZE_PX          = (8, 48)  # Min/max patch size
MEL_BANDS             = 64       # Number of mel bands used
BAND_KEEP_PROB        = 0.3      # Probability each mel band is kept (unmasked)
```

### Checkpoint paths
All checkpoint paths are in `src/utils.py` under `DATASET_CONFIG`. Update these if your checkpoints live somewhere else.

### GPU batch sizes
Also in `src/utils.py`. If you run out of VRAM, reduce these:
```python
MODEL_GPU_BATCH = {
    'resnet': 150,
    'wav2vec2': 64,
    'htsat': 32,   
    'acdnet': 150
}
```

---

## Output

Results are saved to `OUTPUT_DIR` (set by arg 3 or the script default):

```
results/saliency/<model>/<run_name>/
├── <audio_file>_saliency.npy       # Raw saliency map array
├── <audio_file>_saliency.png       # Saliency overlaid on spectrogram
└── evaluation/
    └── evaluation_summary.txt
```

The script skips files that already have a `.npy` output, so interrupted runs can be resumed safely.
