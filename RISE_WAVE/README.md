# RISE_WAVE — Waveform-based Explainability for Audio Classification

Model-agnostic saliency maps for raw audio waveforms using three methods:
**LIME**, **RISE**, and **RISE with Mask & Occlusion Framework**.

Supports two models (**Wav2Vec2**, **ACDNet**) and two datasets (**ESC-50**, **UrbanSound8K**).

---

## Project Structure

```
RISE_WAVE/
├── config/
│   ├── model_configs.py          # Central config: all model + dataset settings
│   └── settings.py
├── src/
│   ├── saliency_scripts/
│   │   ├── saliency_Lime_waveform_unified.py
│   │   ├── Saliency_RISE_waveform_unified.py
│   │   └── Saliency_RISE_waveform_mask_occlusion_framework_unified.py
│   ├── evaluation_scripts/
│   │   └── evaluate_insertion_deletion_waveform.py
│   └── core/
│       └── explanations.py
├── checkpoints/                  # Model weights (see setup below)
├── results/
│   ├── saliency/                 # Generated saliency maps (.npy + .png)
│   └── masks/                    # Pre-generated RISE masks
└── slurm_scripts/
```

---

## Setup

### 1. Checkpoint Paths

Update `config/model_configs.py` → `MODEL_DATASET_OVERRIDES` with your actual checkpoint paths:

| Model     | Dataset      | Key in config                        | Default path                               |
|-----------|--------------|--------------------------------------|--------------------------------------------|
| Wav2Vec2  | ESC-50       | `wav2vec2 / esc50 / weights_path`    | `checkpoints/best_model_wav2vec2_esc50.pt` |
| Wav2Vec2  | UrbanSound8K | `wav2vec2 / urbansound8k / weights_path` | `checkpoints/best_model_wav2vec2_us8k.pt` |
| ACDNet    | ESC-50       | `acdnet / esc50 / weights_path`      | `checkpoints/acdnet_esc50.pt`              |
| ACDNet    | UrbanSound8K | `acdnet / urbansound8k / weights_path` | `checkpoints/acdnet_us8k_best.pt`        |

All scripts are run from the **RISE_WAVE root directory**.

### 2. Audio Data Layout

```
../ESC50/audio/                        # ESC-50 .wav files
../UrbanSound8K/audio/fold10/          # UrbanSound8K .wav files (change fold as needed)
```

---

## Model & Dataset Summary

| Setting          | Wav2Vec2 — ESC-50 | Wav2Vec2 — US8K | ACDNet — ESC-50 | ACDNet — US8K |
|------------------|-------------------|-----------------|-----------------|---------------|
| Sample rate      | 16 000 Hz         | 16 000 Hz       | 20 000 Hz       | 20 000 Hz     |
| Target length    | 80 000 (5 s)      | 64 000 (4 s)    | 30 225 (~1.5 s) | 30 225 (~1.5 s) |
| Classes          | 50                | 10              | 50              | 10            |

---

## Method 1 — LIME Waveform

**Script:** `src/saliency_scripts/saliency_Lime_waveform_unified.py`

Segments the waveform, generates random zero-out perturbations, and fits a weighted Ridge regression to estimate per-segment importance.

### Arguments

| Argument       | Default           | Description                              |
|----------------|-------------------|------------------------------------------|
| `--dataset`    | `urbansound8k`    | `esc50` or `urbansound8k`                |
| `--model`      | `wav2vec2`        | `wav2vec2` or `acdnet`                   |
| `--n_segments` | `10`              | Number of segments to divide audio into  |
| `--num_samples`| `1000`            | Number of random perturbations           |
| `--audio_dir`  | *(auto from dataset)* | Override audio directory             |
| `--output_dir` | *(auto-generated)*| Override output directory                |

### Examples

**Wav2Vec2 on ESC-50:**
```bash
python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --n_segments 50 \
    --num_samples 1000
```

**Wav2Vec2 on UrbanSound8K:**
```bash
python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset urbansound8k \
    --model wav2vec2 \
    --n_segments 50 \
    --num_samples 1000
```

**ACDNet on ESC-50:**
```bash
python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset esc50 \
    --model acdnet \
    --n_segments 50 \
    --num_samples 1000
```

**ACDNet on UrbanSound8K:**
```bash
python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset urbansound8k \
    --model acdnet \
    --n_segments 50 \
    --num_samples 1000
```

### Output

```
results/saliency/saliency_Lime_{model}_{dataset}_{n_segments}_segments/
    {filename}_lime_{model}.npy       # Saliency map [target_length]
    {filename}_lime_{model}.png       # Waveform + heatmap overlay
    {filename}_segment_importance_{model}.png  # Bar chart of segment scores
```

---

## Method 2 — RISE Waveform

**Script:** `src/saliency_scripts/Saliency_RISE_waveform_unified.py`

Original RISE methodology (Petsiuk et al., 2018) adapted for raw audio. Generates random binary masks on a coarse temporal grid, upsamples with smooth interpolation, blends multiplicatively with the input, and accumulates a weighted saliency map.

### Arguments

| Argument        | Default           | Description                                         |
|-----------------|-------------------|-----------------------------------------------------|
| `--dataset`     | `urbansound8k`    | `esc50` or `urbansound8k`                           |
| `--model`       | `wav2vec2`        | `wav2vec2` or `acdnet`                              |
| `--N`           | `6000`            | Number of masks                                     |
| `--n_segments`  | `100`             | Coarse grid resolution (segments)                   |
| `--p1`          | `0.1`             | Probability of keeping each segment                 |
| `--soft_masking`| `linear`          | Upsampling method: `linear`, `step`, `gaussian`     |
| `--edge_sigma`  | `2.0`             | Gaussian smoothing sigma (in segments)              |
| `--occlusion`   | `zeros`           | Occlusion baseline: `zeros` or `gaussian`           |
| `--gpu_batch`   | `50`              | Batch size for GPU inference                        |
| `--generate_new`| *(flag)*          | Force regeneration of masks                         |
| `--audio_dir`   | *(auto from dataset)* | Override audio directory                        |
| `--output_dir`  | *(auto-generated)*| Override output directory                           |

### Examples

**Wav2Vec2 on ESC-50:**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --N 6000 \
    --n_segments 100 \
    --p1 0.1 \
    --soft_masking linear
```

**Wav2Vec2 on UrbanSound8K:**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset urbansound8k \
    --model wav2vec2 \
    --N 6000 \
    --n_segments 100 \
    --p1 0.1 \
    --soft_masking linear
```

**ACDNet on ESC-50:**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset esc50 \
    --model acdnet \
    --N 6000 \
    --n_segments 100 \
    --p1 0.1 \
    --soft_masking linear
```

**ACDNet on UrbanSound8K:**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset urbansound8k \
    --model acdnet \
    --N 6000 \
    --n_segments 100 \
    --p1 0.1 \
    --soft_masking linear
```

### Output

```
results/saliency/saliency_RISE_{model}_{dataset}_{n_segments}_segments/
    {filename}_rise_{model}.npy       # Saliency map [target_length]
    {filename}_rise_{model}.png       # Waveform + heatmap overlay
results/masks/
    masks_rise_{model}_{n_segments}seg_{soft_masking}.npy   # Reusable masks
```

> **Tip:** Masks are saved and reloaded automatically on subsequent runs. Use `--generate_new` to force regeneration.

---

## Method 3 — RISE Waveform Mask & Occlusion Framework

**Script:** `src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py`

Enhanced RISE with structured temporal masks (contiguous segments and scattered patches) and multiple soft-masking / occlusion strategies.

### Arguments

| Argument               | Default           | Description                                              |
|------------------------|-------------------|----------------------------------------------------------|
| `--dataset`            | `urbansound8k`    | `esc50` or `urbansound8k`                                |
| `--model`              | `wav2vec2`        | `wav2vec2` or `acdnet`                                   |
| `--N`                  | `6000`            | Total number of masks                                    |
| `--mask_type`          | `all`             | `all` (mixed), `contiguous`, or `scattered`              |
| `--contiguous_frac`    | `0.5`             | Fraction of contiguous segment masks                     |
| `--scattered_frac`     | `0.5`             | Fraction of scattered patch masks                        |
| `--segment_duration_ms`| `10,500`          | Min,max duration of contiguous segments (ms)             |
| `--patch_duration_ms`  | `5,50`            | Min,max duration of scattered patches (ms)               |
| `--patch_count_range`  | `1,10`            | Min,max number of patches per mask                       |
| `--soft_masking`       | `discrete`        | `discrete`, `gaussian`, or `bilinear`                    |
| `--edge_sigma_ms`      | `10.0`            | Gaussian smoothing sigma (ms)                            |
| `--occlusion`          | `zeros`           | Occlusion baseline: `zeros` or `gaussian`                |
| `--gpu_batch`          | `50`              | Batch size for GPU inference                             |
| `--checkpoint_path`    | *(auto from config)* | Override checkpoint path                              |
| `--target_length`      | *(auto from config)* | Override audio length in samples                      |
| `--sample_rate`        | *(auto from config)* | Override sample rate                                  |
| `--generate_new`       | *(flag)*          | Force regeneration of masks                              |
| `--audio_dir`          | *(auto from dataset)* | Override audio directory                            |
| `--output_dir`         | *(auto-generated)*| Override output directory                                |

### Examples

**Wav2Vec2 on ESC-50 (mixed masks, Gaussian soft masking):**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --N 6000 \
    --mask_type all \
    --soft_masking gaussian \
    --edge_sigma_ms 10.0 \
    --occlusion zeros
```

**Wav2Vec2 on UrbanSound8K (contiguous masks only):**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset urbansound8k \
    --model wav2vec2 \
    --N 6000 \
    --mask_type contiguous \
    --soft_masking discrete \
    --occlusion zeros
```

**ACDNet on ESC-50 (scattered masks, Gaussian noise occlusion):**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset esc50 \
    --model acdnet \
    --N 6000 \
    --mask_type scattered \
    --soft_masking gaussian \
    --occlusion gaussian
```

**ACDNet on UrbanSound8K (mixed masks, bilinear soft masking):**
```bash
python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset urbansound8k \
    --model acdnet \
    --N 6000 \
    --mask_type all \
    --soft_masking bilinear \
    --occlusion zeros
```

### Output

```
results/saliency/saliency_RISE_waveform_{model}_{dataset}_{mask_suffix}{soft_suffix}_occlusion_{occlusion}/
    {filename}_rise_{model}.npy       # Saliency map [target_length]
    {filename}_rise_{model}.png       # Waveform + heatmap overlay
results/masks/
    masks_waveform_{model}_{mask_suffix}{soft_suffix}.npy   # Reusable masks
```

---

## Evaluation — Insertion & Deletion AUC

**Script:** `src/evaluation_scripts/evaluate_insertion_deletion_waveform.py`

Computes Insertion and Deletion AUC scores (Petsiuk et al., 2018) for any saliency method. Progressively inserts (or deletes) the most salient samples and tracks model confidence.

### Arguments

| Argument       | Required | Default        | Description                                                   |
|----------------|----------|----------------|---------------------------------------------------------------|
| `--dataset`    |          | `urbansound8k` | `esc50` or `urbansound8k`                                     |
| `--model`      |          | `wav2vec2`     | `wav2vec2` or `acdnet`                                        |
| `--method`     |          | `rise`         | `lime`, `rise`, or `rise_mo` — determines default file suffix |
| `--audio`      | Yes      |                | Directory containing `.wav` audio files                       |
| `--maps_dir`   | Yes      |                | Directory containing `.npy` saliency maps                     |
| `--output_dir` | Yes      |                | Directory to write evaluation results                         |
| `--suffix`     |          | *(auto)*       | Override saliency file suffix (e.g. `_rise_wav2vec2.npy`)     |

The `--method` flag auto-derives the file suffix:
- `lime` → `_lime_{model}.npy`
- `rise` / `rise_mo` → `_rise_{model}.npy`

### Examples

**Evaluate LIME maps — Wav2Vec2 on ESC-50:**
```bash
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model wav2vec2 \
    --method lime \
    --audio ../ESC50/audio \
    --maps_dir results/saliency/saliency_Lime_wav2vec2_esc50_50_segments \
    --output_dir results/evaluations/ESC50_Wav2Vec2_LIME
```

**Evaluate LIME maps — Wav2Vec2 on UrbanSound8K:**
```bash
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset urbansound8k \
    --model wav2vec2 \
    --method lime \
    --audio ../UrbanSound8K/audio/fold10 \
    --maps_dir results/saliency/saliency_Lime_wav2vec2_urbansound8k_50_segments \
    --output_dir results/evaluations/US8K_Wav2Vec2_LIME
```

**Evaluate RISE maps — ACDNet on ESC-50:**
```bash
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model acdnet \
    --method rise \
    --audio ../ESC50/audio \
    --maps_dir results/saliency/saliency_RISE_acdnet_esc50_100_segments \
    --output_dir results/evaluations/ESC50_ACDNet_RISE
```

**Evaluate RISE maps — ACDNet on UrbanSound8K:**
```bash
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset urbansound8k \
    --model acdnet \
    --method rise \
    --audio ../UrbanSound8K/audio/fold10 \
    --maps_dir results/saliency/saliency_RISE_acdnet_urbansound8k_100_segments \
    --output_dir results/evaluations/US8K_ACDNet_RISE
```

**Evaluate RISE Mask & Occlusion maps — Wav2Vec2 on UrbanSound8K:**
```bash
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset urbansound8k \
    --model wav2vec2 \
    --method rise_mo \
    --audio ../UrbanSound8K/audio/fold10 \
    --maps_dir results/saliency/saliency_RISE_waveform_wav2vec2_urbansound8k_combined_occlusion_zeros \
    --output_dir results/evaluations/US8K_Wav2Vec2_RISE_MO
```

### Output

```
results/evaluations/{run_name}/
    evaluation_results_{model}.npy       # Full per-file scores dict
    evaluation_summary_{model}.txt       # Mean AUC table + per-file breakdown
    auc_distributions_{model}.png        # Histogram of insertion/deletion AUCs
```

---

## Quick-Reference: All Combinations

| Dataset      | Model     | LIME | RISE | RISE-MO | Evaluation |
|--------------|-----------|------|------|---------|------------|
| ESC-50       | Wav2Vec2  | ✓    | ✓    | ✓       | ✓          |
| ESC-50       | ACDNet    | ✓    | ✓    | ✓       | ✓          |
| UrbanSound8K | Wav2Vec2  | ✓    | ✓    | ✓       | ✓          |
| UrbanSound8K | ACDNet    | ✓    | ✓    | ✓       | ✓          |

---

## Typical Workflow

```bash
# 1. Generate saliency maps with your chosen method
python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset esc50 --model wav2vec2 --N 6000 --n_segments 100

# 2. Evaluate with insertion/deletion
python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 --model wav2vec2 --method rise \
    --audio ../ESC50/audio \
    --maps_dir results/saliency/saliency_RISE_wav2vec2_esc50_100_segments \
    --output_dir results/evaluations/ESC50_Wav2Vec2_RISE
```
