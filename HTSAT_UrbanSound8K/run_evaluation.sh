#!/bin/bash
#SBATCH --job-name=HTSAT_US8K
#SBATCH --partition=gpu
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

# HTSAT Evaluation Script for UrbanSound8K using AudioSet checkpoint
# Based on: https://github.com/RetroCirce/HTS-Audio-Transformer

# Configuration
CHECKPOINT="training_output_fold10/best_model.pth"  # Path to AudioSet checkpoint
AUDIO_DIR="../UrbanSound8K/audio"  # Parent directory containing fold1, fold2, etc.
METADATA="../UrbanSound8K/metadata/UrbanSound8K.csv"  # Metadata file
TEST_FOLD=10  # Test fold (UrbanSound8K uses fold 10 for testing)
BATCH_SIZE=32
NUM_WORKERS=4
DEVICE="cuda"  # Change to "cpu" if no GPU available
OUTPUT_DIR="./results_fold${TEST_FOLD}"

echo "=========================================="
echo "HTSAT AudioSet → UrbanSound8K Evaluation"
echo "=========================================="
echo "AudioSet Checkpoint: $CHECKPOINT"
echo "Audio Directory: $AUDIO_DIR"
echo "Metadata: $METADATA"
echo "Test Fold: $TEST_FOLD"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation
python evaluate_urbansound8k.py \
    --checkpoint "$CHECKPOINT" \
    --audio_dir "$AUDIO_DIR" \
    --metadata "$METADATA" \
    --test_fold $TEST_FOLD \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Evaluation complete! Results saved to $OUTPUT_DIR"

