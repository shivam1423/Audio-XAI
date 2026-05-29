# UrbanSound8K Model Checkpoints

This directory should contain the trained model checkpoints for UrbanSound8K dataset.

## Required Checkpoint Files

Please place the following 4 checkpoint files in this directory:

1. **`best_model_wav2vec2_us8k.pt`** - Wav2Vec2 model trained on UrbanSound8K
2. **`acdnet_us8k_best.pt`** - ACDNet model trained on UrbanSound8K  
3. **`resnet50_urbansound8k.pt`** - ResNet50 model trained on UrbanSound8K
4. **`htsat_us8k.pth`** - HTSAT model trained on UrbanSound8K

## Expected Model Specifications

- **Wav2Vec2**: 10 classes, 16kHz sample rate, 64000 samples (4 seconds)
- **ACDNet**: 10 classes, 20kHz sample rate, 30225 samples (~1.5 seconds)
- **ResNet50**: 10 classes, spectrogram-based
- **HTSAT**: 10 classes, 32kHz sample rate, waveform or spectrogram input

## Usage

Once all checkpoint files are in place, the saliency scripts will automatically load them:

```bash
# Waveform scripts (LIME, RISE)
python src/saliency_scripts/saliency_Lime_waveform_unified.py --model wav2vec2 --dataset urbansound8k
python src/saliency_scripts/Saliency_RISE_waveform_unified.py --model acdnet

# Spectrogram scripts (LIME, GradCAM)
python src/saliency_scripts/Saliency_Lime_framework.py --model_type htsat --dataset urbansound8k
python src/saliency_scripts/Saliency_Gradcam.py
```

## Checkpoint Format

### Wav2Vec2
Expected format: `{'model_state_dict': state_dict}` or direct `state_dict`

### ACDNet  
Expected format: `{'model_state_dict': state_dict}` or `{'weight': state_dict}` or direct `state_dict`

### ResNet50
Standard PyTorch state_dict

### HTSAT
HTSAT checkpoint format (`.pth` file)
