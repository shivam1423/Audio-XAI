#!/bin/bash
#SBATCH --job-name=HTSAT_train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=16000M
#SBATCH --time=8-08:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Fine-tune HTSAT on UrbanSound8K
# Transfer learning from AudioSet checkpoint

# Configuration
CHECKPOINT="HTSAT_AudioSet_Saved_1.ckpt"  # Path to AudioSet checkpoint
AUDIO_DIR="../UrbanSound8K/audio"  # Parent directory containing fold1, fold2, etc.
METADATA="../UrbanSound8K/metadata/UrbanSound8K.csv"  # Metadata file
TEST_FOLD=10  # Test fold (UrbanSound8K uses fold 10 for testing)
EPOCHS=30  # Number of training epochs
BATCH_SIZE=16  # Batch size (smaller for training to fit in GPU memory)
LEARNING_RATE=1e-4  # Learning rate
FREEZE_FEATURES=true  # Freeze AudioSet features (faster training)
DEVICE="cuda"  # Change to "cpu" if no GPU available
OUTPUT_DIR="./training_output_fold${TEST_FOLD}"

echo "=========================================="
echo "HTSAT Fine-tuning on UrbanSound8K"
echo "=========================================="
echo "AudioSet Checkpoint: $CHECKPOINT"
echo "Audio Directory: $AUDIO_DIR"
echo "Metadata: $METADATA"
echo "Test Fold: $TEST_FOLD"
echo "Training Folds: 1-9 (excluding $TEST_FOLD)"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Freeze Features: $FREEZE_FEATURES"
echo "Device: $DEVICE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Build command
CMD="python train_urbansound8k.py \
    --checkpoint $CHECKPOINT \
    --audio_dir $AUDIO_DIR \
    --metadata $METADATA \
    --test_fold $TEST_FOLD \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR"

# Add freeze flag if enabled
if [ "$FREEZE_FEATURES" = true ]; then
    CMD="$CMD --freeze_features"
fi

# Run training
echo "Starting training..."
echo ""
$CMD

echo ""
echo "Training complete! Results saved to $OUTPUT_DIR"

