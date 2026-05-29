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

# Run RISE
#srun python src/saliency_scripts/Saliency_Lime_framework.py --model_type resnet50 --dataset esc50 --audio_dir test_audio --use_audio
#
#srun python src/saliency_scripts/Saliency.py --model_type resnet50 --dataset esc50 --audio_dir test_audio --use_audio

#srun python src/saliency_scripts/Saliency_Gradcam.py --model_type resnet50 --dataset esc50 --audio_dir test_audio

srun python src/saliency_scripts/saliency_mask_occlusion_framework.py --model_type resnet50 --dataset esc50 --audio_dir test_audio --use_audio