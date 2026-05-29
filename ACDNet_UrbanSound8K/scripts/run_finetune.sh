#!/bin/bash
#SBATCH --job-name=acdnet_finetune
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=1-12:00:00

# ============================================================================
# ACDNet Transfer Learning: Finetune ESC-50 → UrbanSound8K
# ============================================================================
# This script finetunes a pretrained ESC-50 checkpoint on UrbanSound8K
# Expected benefits:
#   - Faster convergence (3-5 hours vs 15-20 hours)
#   - Better accuracy (85-88% vs 82-85%)
#   - Learned features from 50 diverse ESC-50 classes
# ============================================================================

# Load modules (uncomment if needed on your cluster)
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Display GPU information
nvidia-smi
echo -e "\nNode: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"
echo ""

# ============================================================================
# Configuration
# ============================================================================
# IMPORTANT: Update these paths for your setup
PRETRAINED_CHECKPOINT="../ACDNet/acdnet_weight_pruned_trained_fold4_90.50.pt"  # ESC-50 checkpoint
DATA_DIR="../UrbanSound8K"     # UrbanSound8K dataset location
NPZ_DIR="./data"                # Directory containing preprocessed NPZ
NPZ_FILE="$NPZ_DIR/urbansound8k_20k.npz"  # Preprocessed NPZ file
VAL_DATA_DIR="./val_data"       # Multi-crop validation data directory
OUTPUT_DIR="./finetune_models"  # Output directory for finetuned models
DEVICE="cuda"

# Finetuning hyperparameters (optimized for transfer learning)
BATCH_SIZE=64
EPOCHS=200                      # Fewer than from-scratch (500)
LEARNING_RATE=0.01              # 10x smaller than from-scratch (0.1)
SEED=42

# Optional: Freeze SFEB layers for faster finetuning
# Set to "true" to freeze, "false" to finetune all layers
FREEZE_SFEB=false

echo "=========================================="
echo "ACDNet Transfer Learning (Finetuning)"
echo "=========================================="
echo "Pretrained Checkpoint: $PRETRAINED_CHECKPOINT"
echo "NPZ File: $NPZ_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Freeze SFEB: $FREEZE_SFEB"
echo "Random Seed: $SEED"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Verify pretrained checkpoint exists
# ============================================================================
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT"
    echo "Please update PRETRAINED_CHECKPOINT in this script"
    exit 1
fi

echo "✓ Pretrained checkpoint found: $PRETRAINED_CHECKPOINT"
echo ""

# ============================================================================
# Step 1: Check and preprocess dataset if needed
# ============================================================================
if [ ! -f "$NPZ_FILE" ]; then
    echo "=========================================="
    echo "NPZ file not found. Preprocessing dataset..."
    echo "=========================================="
    echo ""
    
    # Verify data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        echo "Please update DATA_DIR in this script"
        exit 1
    fi
    
    # Verify dataset structure
    if [ ! -d "$DATA_DIR/audio/fold1" ]; then
        echo "ERROR: Expected structure not found: $DATA_DIR/audio/fold1/"
        echo "UrbanSound8K should have: $DATA_DIR/audio/fold1/, fold2/, ..., fold10/"
        exit 1
    fi
    
    # Create NPZ directory
    mkdir -p $NPZ_DIR
    
    # Run preprocessing
    echo "Preprocessing UrbanSound8K to NPZ format..."
    echo "This will take 30-60 minutes but is only needed ONCE."
    echo ""
    
    srun python -u scripts/prepare_urbansound8k.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$NPZ_DIR" \
        --sr 20000 \
        --verify
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Preprocessing failed"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "NPZ file created: $NPZ_FILE"
    echo "=========================================="
    echo ""
else
    echo "✓ NPZ file found: $NPZ_FILE"
    echo "  Skipping preprocessing (already done)"
    echo ""
fi

# ============================================================================
# Step 1.5: Prepare multi-crop validation data if needed
# ============================================================================
if [ ! -f "$VAL_DATA_DIR/fold9_val10crop.npz" ] || [ ! -f "$VAL_DATA_DIR/fold10_val10crop.npz" ]; then
    echo "=========================================="
    echo "Preparing multi-crop validation data..."
    echo "=========================================="
    echo ""
    
    # Create validation data directory
    mkdir -p $VAL_DATA_DIR
    
    # Run multi-crop preprocessing
    echo "Generating 10 evenly-spaced crops per sample for validation/test..."
    echo "This follows original ACDNet methodology."
    echo ""
    
    srun python -u scripts/prepare_validation_data.py \
        --npz_path "$NPZ_FILE" \
        --output_dir "$VAL_DATA_DIR" \
        --val_fold 9 \
        --test_fold 10 \
        --input_length 30225 \
        --n_crops 10
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Validation data preparation failed"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Multi-crop validation data created!"
    echo "=========================================="
    echo ""
else
    echo "✓ Multi-crop validation data found:"
    echo "  - $VAL_DATA_DIR/fold9_val10crop.npz"
    echo "  - $VAL_DATA_DIR/fold10_val10crop.npz"
    echo "  Skipping multi-crop preparation (already done)"
    echo ""
fi

# ============================================================================
# Step 2: Finetune model with pretrained checkpoint
# ============================================================================
# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Starting finetuning..."
echo "=========================================="
echo ""

# Build freeze argument
FREEZE_ARG=""
if [ "$FREEZE_SFEB" = "true" ]; then
    FREEZE_ARG="--freeze_sfeb"
fi

srun python -u scripts/finetune.py \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --npz_path "$NPZ_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --seed $SEED \
    $FREEZE_ARG

# ============================================================================
# Step 3: Check training result and run evaluation
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Finetuning completed successfully!"
    echo "=========================================="
    echo ""
    
    # Run evaluation on test set
    echo "=========================================="
    echo "Starting evaluation on test set..."
    echo "=========================================="
    echo ""
    
    srun python -u scripts/evaluate.py \
        --npz_path "$NPZ_FILE" \
        --model_path "$OUTPUT_DIR/acdnet_us8k_best.pt" \
        --output_dir "$OUTPUT_DIR/results" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Transfer learning completed successfully!"
        echo "=========================================="
        echo "Finetuned model saved to: $OUTPUT_DIR/acdnet_us8k_best.pt"
        echo "Results saved to: $OUTPUT_DIR/results"
        echo ""
        echo "Expected improvements:"
        echo "  - Training time: 3-5 hours (vs 15-20 hours from scratch)"
        echo "  - Convergence: Faster (good accuracy by epoch 50)"
        echo "  - Final accuracy: 85-88% (vs 82-85% from scratch)"
        echo "=========================================="
    else
        echo ""
        echo "ERROR: Evaluation failed"
        exit 1
    fi
else
    echo ""
    echo "ERROR: Finetuning failed"
    exit 1
fi
