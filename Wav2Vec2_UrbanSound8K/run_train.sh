#!/bin/bash
#SBATCH --job-name=wav2vec2_us8k
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
srun python train.py \
    --data_dir ../UrbanSound8K \
    --batch_size 16 \
    --num_epochs 30 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --sample_rate 16000 \
    --max_duration 4.0 \
    --num_workers 4 \
    --train_folds 1 2 3 4 5 6 7 8 \
    --val_fold 9 \
    --test_fold 10 \
    --output_dir outputs \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --seed 42

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
