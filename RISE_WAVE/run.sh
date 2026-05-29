#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=ifn
#SBATCH --partition=ifn
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1080:2


# RISE Waveform Explanation - Run Script
# Supports: wav2vec2, acdnet
# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo "Node: $(hostname)"
echo "======================================"

# Run all three saliency methods on a single ESC-50 test audio file
TEST_AUDIO="test_audio"

# RISE (uniform segment masks)
srun python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --audio_dir "$TEST_AUDIO" \
    --n_segments 100 \
    --N 6000 \
    --occlusion zeros \
    --output_dir results/test/rise_wav2vec2

srun python src/saliency_scripts/Saliency_RISE_waveform_unified.py \
    --dataset esc50 \
    --model acdnet \
    --audio_dir "$TEST_AUDIO" \
    --n_segments 100 \
    --N 6000 \
    --occlusion zeros \
    --output_dir results/test/rise_acdnet

# RISE with mask-occlusion framework
srun python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --audio_dir "$TEST_AUDIO" \
    --mask_type all \
    --N 6000 \
    --occlusion zeros \
    --output_dir results/test/rise_mo_wav2vec2

srun python src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework_unified.py \
    --dataset esc50 \
    --model acdnet \
    --audio_dir "$TEST_AUDIO" \
    --mask_type all \
    --N 6000 \
    --occlusion zeros \
    --output_dir results/test/rise_mo_acdnet

# LIME
srun python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset esc50 \
    --model wav2vec2 \
    --audio_dir "$TEST_AUDIO" \
    --n_segments 100 \
    --num_samples 1000 \
    --output_dir results/test/lime_wav2vec2

srun python src/saliency_scripts/saliency_Lime_waveform_unified.py \
    --dataset esc50 \
    --model acdnet \
    --audio_dir "$TEST_AUDIO" \
    --n_segments 100 \
    --num_samples 1000 \
    --output_dir results/test/lime_acdnet