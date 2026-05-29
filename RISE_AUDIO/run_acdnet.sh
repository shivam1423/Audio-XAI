#!/bin/bash
#SBATCH --job-name=riseaudio_waveform
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=25000M
#SBATCH --time=8-08:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

echo "======================================"
echo "RISE Audio - ACDNET"
echo "======================================"

MODEL_TYPE=${1:-'acdnet'}
INPUT_DIR=${2:-'../ESC50/audio'}
OUTPUT_DIR=${3:-"results/saliency/acdnet/ESC50_48khz_all_multiple_6000masks_discrete"}
SOFT_MASKING=${4:-'none'}
OCCLUSION=${5:-'black'}

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
echo "✓ ACDNET processing complete!"
echo "======================================"
echo "Results saved to: results/saliency/"
echo "Evaluation results: results/saliency/.../evaluation/"

