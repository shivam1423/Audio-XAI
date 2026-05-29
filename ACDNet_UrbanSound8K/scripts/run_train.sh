#!/bin/bash
#SBATCH --job-name=acdnet_us8k
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=8-08:00:00

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
# IMPORTANT: Update DATA_DIR to point to your UrbanSound8K dataset location
DATA_DIR="../UrbanSound8K"  # Path to UrbanSound8K dataset (contains metadata/ and audio/ folders)
NPZ_DIR="./data"             # Directory to store preprocessed NPZ file
NPZ_FILE="$NPZ_DIR/urbansound8k_20k.npz"  # Preprocessed NPZ file path
VAL_DATA_DIR="./val_data"    # Directory for multi-crop validation data
OUTPUT_DIR="./trained_models"
RESULTS_DIR="./results"
DEVICE="cuda"
BATCH_SIZE=64                # Updated to match original ACDNet
EPOCHS=500                   # Updated to match original ACDNet pattern
LEARNING_RATE=0.1
SEED=42
VAL_FOLD=9
TEST_FOLD=10

echo "=========================================="
echo "ACDNet Training on UrbanSound8K (NPZ)"
echo "=========================================="
echo "Data Directory: $DATA_DIR"
echo "NPZ File: $NPZ_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Random Seed: $SEED"
echo "=========================================="
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
        echo "Please update DATA_DIR in this script to point to your UrbanSound8K dataset"
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
    
    # Run preprocessing (takes ~30-60 minutes, but only needed once!)
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
if [ ! -f "$VAL_DATA_DIR/fold${VAL_FOLD}_val10crop.npz" ] || [ ! -f "$VAL_DATA_DIR/fold${TEST_FOLD}_val10crop.npz" ]; then
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
        --val_fold $VAL_FOLD \
        --test_fold $TEST_FOLD \
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
    echo "  - $VAL_DATA_DIR/fold${VAL_FOLD}_val10crop.npz"
    echo "  - $VAL_DATA_DIR/fold${TEST_FOLD}_val10crop.npz"
    echo "  Skipping multi-crop preparation (already done)"
    echo ""
fi

# ============================================================================
# Step 2: Train model
# ============================================================================
# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $RESULTS_DIR

echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

srun python -u scripts/train.py \
    --npz_path "$NPZ_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --seed $SEED

# ============================================================================
# Step 3: Check training result and run evaluation
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
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
        --output_dir "$RESULTS_DIR" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ All tasks completed successfully!"
        echo "=========================================="
        echo "Model saved to: $OUTPUT_DIR/acdnet_us8k_best.pt"
        echo "Results saved to: $RESULTS_DIR"
        echo "=========================================="
    else
        echo ""
        echo "ERROR: Evaluation failed"
        exit 1
    fi
else
    echo ""
    echo "ERROR: Training failed"
    exit 1
fi
