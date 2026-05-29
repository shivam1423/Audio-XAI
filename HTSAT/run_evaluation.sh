#!/bin/bash
#SBATCH --job-name=WavJEPA
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=2000M
#SBATCH --time=8-08:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"
# HTSAT Evaluation Script for ESC-50
# Based on: https://github.com/RetroCirce/HTS-Audio-Transformer

# Configuration
CHECKPOINT="HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
AUDIO_DIR="../ESC50/audio"
VAL_FOLD=2  # Validation fold
BATCH_SIZE=32
NUM_WORKERS=4
DEVICE="cuda"  # Change to "cpu" if no GPU available
OUTPUT_DIR="./results_fold${VAL_FOLD}"

echo "=========================================="
echo "HTSAT Evaluation on ESC-50"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Audio Directory: $AUDIO_DIR"
echo "Validation Fold: $VAL_FOLD"
echo "Training Folds: 1, 3, 4, 5"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation using official HTSAT architecture
python evaluate_htsat.py \
    --checkpoint "$CHECKPOINT" \
    --audio_dir "$AUDIO_DIR" \
    --val_fold $VAL_FOLD \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Evaluation complete! Results saved to $OUTPUT_DIR"

