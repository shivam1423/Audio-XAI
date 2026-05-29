#!/bin/bash
#SBATCH --job-name=resnet50_us8k
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# ResNet50 UrbanSound8K Training Script
# This script trains ResNet50 model on UrbanSound8K dataset using SLURM

echo "=========================================="
echo "ResNet50 UrbanSound8K Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Display GPU information
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi
    echo ""
fi

# Display environment information
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""

# Configuration
DATA_DIR="../UrbanSound8K"
OUTPUT_DIR="./trained_models"
DEVICE="cuda"
BATCH_SIZE=32
EPOCHS=100
LR=0.001
OPTIMIZER="sgd"
NUM_WORKERS=4
SEED=42

echo "Training Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Optimizer: $OPTIMIZER"
echo "  Num workers: $NUM_WORKERS"
echo "  Random seed: $SEED"
echo "=========================================="
echo ""

# Execute training
python scripts/train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer "$OPTIMIZER" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
