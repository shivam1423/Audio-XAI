#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --partition=gpub
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=8-08:00:00

# Get parameters
MASK_TYPE=$1
SOFT_MASKING=$2

echo "============================================="
echo "Starting RISE saliency generation"
echo "Mask Type: $MASK_TYPE"
echo "Soft Masking: $SOFT_MASKING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "============================================="

# Load modules (uncomment if needed)
#module load anaconda/3-5.0.1
#module load cuda/11.1

# GPU information
nvidia-smi
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Execute saliency generation
echo "Running saliency generation..."
srun python saliency_masking_strategy.py \
    --mask_type "$MASK_TYPE" \
    --soft_masking "$SOFT_MASKING" \
    --datadir ESC50_spectrograms/

echo "Saliency generation completed for mask_type=$MASK_TYPE, soft_masking=$SOFT_MASKING"
