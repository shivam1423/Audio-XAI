#!/bin/bash
#SBATCH --job-name=wav2vec2_us8k_eval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# UrbanSound8K Evaluation Script for Wav2Vec2
# This script evaluates a trained Wav2Vec2 model on UrbanSound8K test set

echo "=========================================="
echo "Wav2Vec2 UrbanSound8K Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.8
# module load cuda/11.3

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "=========================================="
fi

# Run evaluation
python scripts/evaluate.py \
       --npz_path ./data/urbansound8k_20k.npz \
       --model_path ./trained_models/acdnet_us8k_best.pt \
       --output_dir ./results \
       --device cuda

echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="
