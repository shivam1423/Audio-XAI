#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=ifn
#SBATCH --partition=ifn
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1080:1


# RISE Waveform Explanation - Run Script
# Supports: wav2vec2, acdnet

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

echo "======================================"
echo "RISE Audio - Wav2Vec2"
echo "======================================"

MODEL_TYPE=${1:-'wav2vec2'}
INPUT_DIR=${2:-'../ESC50/audio_esc10'}
OUTPUT_DIR=${3:-"results/saliency/wav2vec2/ESC10_48khz_all_multiple_6000masks_discrete"}
SOFT_MASKING=${4:-'none'}
OCCLUSION=${5:-'freq'}

echo "Configuration:"
echo "  Model: $MODEL_TYPE"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Soft Masking: $SOFT_MASKING"
echo "  Occlusion: $OCCLUSION"

export MODEL_TYPE
export INPUT_DIR
export OUTPUT_DIR
export SOFT_MASKING
export OCCLUSION

# Run saliency generation
echo ""
echo "Starting saliency generation..."
srun python -u main.py

echo ""
echo "======================================"
echo "✓ Wav2Vec2 processing complete!"
echo "======================================"
echo "Results saved to: results/saliency/"
echo "Evaluation results: results/saliency/.../evaluation/"

