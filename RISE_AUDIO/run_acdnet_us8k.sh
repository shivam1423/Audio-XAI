#!/bin/bash
#SBATCH --job-name=acdnet_riseaudio_waveform
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=40000M
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
# Dataset selection: 'esc50' or 'urbansound8k'
DATASET=${1:-'urbansound8k'}
MODEL_TYPE='acdnet'

# Set default paths based on dataset
if [ "$DATASET" = "urbansound8k" ]; then
    DEFAULT_INPUT_DIR='../UrbanSound8K/audio/fold10' #'test_audio/'
    DEFAULT_OUTPUT_DIR="results/saliency/acdnet/UrbanSound8K_fold10_48khz_all_multiple_6000masks_discrete"
else
    DEFAULT_INPUT_DIR='../ESC50/audio'
    DEFAULT_OUTPUT_DIR="results/saliency/acdnet/ESC50_48khz_all_multiple_6000masks_discrete"
fi


MODEL_TYPE=${1:-'acdnet'}
INPUT_DIR=${2:-$DEFAULT_INPUT_DIR}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}
SOFT_MASKING=${4:-'none'}
OCCLUSION=${5:-'black'}

echo "Configuration:"
echo "  Model: $MODEL_TYPE"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Soft Masking: $SOFT_MASKING"
echo "  Occlusion: $OCCLUSION"

export DATASET
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

