#!/bin/bash
#SBATCH --job-name=acdnet_us8k
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=8-08:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"


# UrbanSound8K Training Script for Wav2Vec2
# This script trains Wav2Vec2 model on UrbanSound8K dataset

echo "=========================================="
echo "Wav2Vec2 UrbanSound8K Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="


# Execute programs
srun python scripts/train.py \
    --data_dir ../UrbanSound8K \
    --output_dir ./trained_models \
    --device cpu \
    --batch_size 32 \
    --epochs 120 \
    --lr 0.1 \
    --seed 42

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
