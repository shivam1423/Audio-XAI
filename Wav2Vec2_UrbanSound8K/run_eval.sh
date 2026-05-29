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
#srun python train.py \
#    --data_dir ../UrbanSound8K \
#    --test_only \
#    --resume checkpoints/best_model_wav2vec2_us8k.pt \
#    --batch_size 16 \
#    --num_workers 4 \
#    --test_fold 10
srun python inference.py \
    --checkpoint checkpoints/best_model_wav2vec2_us8k.pt \
    --audio_dir ../UrbanSound8K/audio/fold10 \
    --output_file predictions_fold10.csv \
    --device auto
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="
