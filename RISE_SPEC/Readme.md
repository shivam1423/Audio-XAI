# Running Saliency Scripts

All commands must be run from the **`RISE_SPEC/`** project root.

---

## Prerequisites

### 1. Checkpoint files

Place checkpoint files in `checkpoints/` following the naming expected by `config/model_configs.py`:

| Model | Dataset | Expected filename |
|---|---|---|
| ResNet50 | ESC-50 | `checkpoints/resnet50_esc50.pth` |
| ResNet50 | UrbanSound8K | `checkpoints/resnet50_urbansound8k.pth` |
| HTSAT | ESC-50 | `checkpoints/htsat_esc50.ckpt` |
| HTSAT | UrbanSound8K | `checkpoints/htsat_us8k.pth` |

If your filenames differ, pass `--weights_path /path/to/checkpoint` on any command below.

### 2. Audio data layout

Default paths (relative to `RISE_SPEC/`):

| Dataset | Default audio directory |
|---|---|
| ESC-50 | `../ESC50/audio` |
| UrbanSound8K | `../UrbanSound8K/audio/fold10` |

Override with `--audio_dir /your/path` if your data is elsewhere.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Saliency Scripts

### 1. RISE — `src/saliency_scripts/Saliency.py`

Supports: ResNet50, HTSAT × ESC-50, UrbanSound8K

```bash
# ResNet50 — ESC-50
python src/saliency_scripts/Saliency.py \
    --model_type resnet50 --dataset esc50

# ResNet50 — UrbanSound8K
python src/saliency_scripts/Saliency.py \
    --model_type resnet50 --dataset urbansound8k

# HTSAT — ESC-50
python src/saliency_scripts/Saliency.py \
    --model_type htsat --dataset esc50

# HTSAT — UrbanSound8K
python src/saliency_scripts/Saliency.py \
    --model_type htsat --dataset urbansound8k
```

Output: `results/{model_type}/saliency/{dataset}_saliency_maps/`
Saliency file suffix: `_saliency.npy`

---

### 2. Grad-CAM — `src/saliency_scripts/Saliency_Gradcam.py`

> **ResNet50 only.** Grad-CAM requires a CNN backbone with spatial feature maps.

```bash
# ResNet50 — ESC-50
python src/saliency_scripts/Saliency_Gradcam.py \
    --model_type resnet50 --dataset esc50

# ResNet50 — UrbanSound8K
python src/saliency_scripts/Saliency_Gradcam.py \
    --model_type resnet50 --dataset urbansound8k
```

Output: `results/saliency/gradcam_resnet50_{dataset}/`
Saliency file suffix: `_gradcam.npy`

---

### 3. LIME — `src/saliency_scripts/Saliency_Lime_framework.py`

Supports: ResNet50, HTSAT × ESC-50, UrbanSound8K

```bash
# ResNet50 — ESC-50
python src/saliency_scripts/Saliency_Lime_framework.py \
    --model_type resnet50 --dataset esc50

# ResNet50 — UrbanSound8K
python src/saliency_scripts/Saliency_Lime_framework.py \
    --model_type resnet50 --dataset urbansound8k

# HTSAT — ESC-50
python src/saliency_scripts/Saliency_Lime_framework.py \
    --model_type htsat --dataset esc50

# HTSAT — UrbanSound8K
python src/saliency_scripts/Saliency_Lime_framework.py \
    --model_type htsat --dataset urbansound8k
```

Output: `results/{model_type}/saliency/saliency_lime_{dataset}/`
Saliency file suffix: `_lime.npy`

Optional: `--num_samples 2000` (default 1000) increases explanation quality at the cost of runtime.

---

### 4. TF-Structured RISE — `src/saliency_scripts/saliency_mask_occlusion_framework.py`

Supports: ResNet50, HTSAT × ESC-50, UrbanSound8K

```bash
# ResNet50 — ESC-50
python src/saliency_scripts/saliency_mask_occlusion_framework.py \
    --model_type resnet50 --dataset esc50 \
    --mask_type all --soft_masking bilinear --occlusion black

# ResNet50 — UrbanSound8K
python src/saliency_scripts/saliency_mask_occlusion_framework.py \
    --model_type resnet50 --dataset urbansound8k \
    --mask_type all --soft_masking bilinear --occlusion black

# HTSAT — ESC-50
python src/saliency_scripts/saliency_mask_occlusion_framework.py \
    --model_type htsat --dataset esc50 \
    --mask_type all --soft_masking bilinear --occlusion freq

# HTSAT — UrbanSound8K
python src/saliency_scripts/saliency_mask_occlusion_framework.py \
    --model_type htsat --dataset urbansound8k \
    --mask_type all --soft_masking bilinear --occlusion freq
```

Output: `results/{model_type}/saliency/{dataset}_all_bilinear_occlusion_{occlusion}/`
Saliency file suffix: `_saliency.npy`

Key options:

| Flag | Choices | Description |
|---|---|---|
| `--mask_type` | `all`, `time`, `freq`, `rect`, `mel` | Mask strategy; `all` mixes all four |
| `--soft_masking` | `bilinear`, `gaussian`, `none` | Edge softening method |
| `--occlusion` | `black`, `time`, `freq` | Baseline for masked regions; use `freq` for HTSAT |
| `--N` | integer | Number of masks (default 6000) |
| `--generate_new` | flag | Force regeneration even if mask file exists |

---

### Common overrides (all scripts)

```bash
--audio_dir /path/to/audio     # override default audio directory
--weights_path /path/to/ckpt   # override default checkpoint
--gpu_batch 100                # reduce if you hit GPU OOM
```

---

## Evaluation — Insertion / Deletion AUC

Script: `src/evaluation_scripts/evaluate_insertion_deletion.py`

Arguments:

| Flag | Description |
|---|---|
| `--images` | Directory containing the audio files used during saliency generation |
| `--maps_dir` | Directory containing the per-file saliency `.npy` files |
| `--suffix` | Filename suffix of saliency files (default: `_saliency.npy`) |
| `--output_dir` | Where to write results |
| `--model_type` | `resnet50` or `htsat` |
| `--dataset` | `esc50` or `urbansound8k` |
| `--use_audio` | Pass this flag when saliency was generated from raw audio |

---

### RISE

```bash
# ResNet50 — UrbanSound8K
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../UrbanSound8K/audio/fold10 \
    --maps_dir results/resnet50/saliency/urbansound8k_saliency_maps \
    --suffix _saliency.npy \
    --output_dir results/resnet50/eval/rise_urbansound8k \
    --model_type resnet50 --dataset urbansound8k --use_audio

# HTSAT — ESC-50
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../ESC50/audio \
    --maps_dir results/htsat/saliency/esc50_saliency_maps \
    --suffix _saliency.npy \
    --output_dir results/htsat/eval/rise_esc50 \
    --model_type htsat --dataset esc50 --use_audio
```

---

### Grad-CAM (ResNet50 only)

```bash
# ResNet50 — UrbanSound8K
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../UrbanSound8K/audio/fold10 \
    --maps_dir results/saliency/gradcam_resnet50_urbansound8k \
    --suffix _gradcam.npy \
    --output_dir results/resnet50/eval/gradcam_urbansound8k \
    --model_type resnet50 --dataset urbansound8k --use_audio

# ResNet50 — ESC-50
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../ESC50/audio \
    --maps_dir results/saliency/gradcam_resnet50_esc50 \
    --suffix _gradcam.npy \
    --output_dir results/resnet50/eval/gradcam_esc50 \
    --model_type resnet50 --dataset esc50 --use_audio
```

---

### LIME

```bash
# ResNet50 — UrbanSound8K
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../UrbanSound8K/audio/fold10 \
    --maps_dir results/resnet50/saliency/saliency_lime_urbansound8k \
    --suffix _lime.npy \
    --output_dir results/resnet50/eval/lime_urbansound8k \
    --model_type resnet50 --dataset urbansound8k --use_audio

# HTSAT — ESC-50
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../ESC50/audio \
    --maps_dir results/htsat/saliency/saliency_lime_esc50 \
    --suffix _lime.npy \
    --output_dir results/htsat/eval/lime_esc50 \
    --model_type htsat --dataset esc50 --use_audio
```

---

### TF-Structured RISE

```bash
# ResNet50 — UrbanSound8K
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../UrbanSound8K/audio/fold10 \
    --maps_dir results/resnet50/saliency/urbansound8k_all_bilinear_occlusion_black \
    --suffix _saliency.npy \
    --output_dir results/resnet50/eval/tf_rise_urbansound8k \
    --model_type resnet50 --dataset urbansound8k --use_audio

# HTSAT — ESC-50
python src/evaluation_scripts/evaluate_insertion_deletion.py \
    --images ../ESC50/audio \
    --maps_dir results/htsat/saliency/esc50_all_bilinear_occlusion_freq \
    --suffix _saliency.npy \
    --output_dir results/htsat/eval/tf_rise_esc50 \
    --model_type htsat --dataset esc50 --use_audio
```

---

## Results layout after a full run

```
results/
├── resnet50/
│   ├── saliency/
│   │   ├── urbansound8k_saliency_maps/        # RISE
│   │   ├── saliency_lime_urbansound8k/        # LIME
│   │   └── urbansound8k_all_bilinear_.../     # RISE_SPEC
│   └── eval/
│       ├── rise_urbansound8k/
│       ├── gradcam_urbansound8k/
│       ├── lime_urbansound8k/
│       └── tf_rise_urbansound8k/
├── htsat/
│   ├── saliency/
│   │   ├── esc50_saliency_maps/
│   │   └── ...
│   └── eval/
│       └── ...
└── saliency/
    └── gradcam_resnet50_{dataset}/            # GradCAM (separate root)
```
