#!/bin/bash
#SBATCH --job-name=eval_batch_4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=10000M
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/eval_batch_4_%j.out
#SBATCH --error=logs/eval_batch_4_%j.err

# Load modules (uncomment if needed)
#module load anaconda/3-5.0.1
#module load cuda/11.1

echo "=== Starting Evaluation Batch 4 ==="
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Function to run single evaluation
run_evaluation() {
    local mask_type=$1
    local soft_masking=$2

    echo "----------------------------------------"
    echo "Running: mask_type=$mask_type, soft_masking=$soft_masking"
    echo "----------------------------------------"

    # Handle directory naming
    if [ "$soft_masking" = "none" ]; then
        SOFT_MASKING_DIR="discrete"
    else
        SOFT_MASKING_DIR="$soft_masking"
    fi

    if [ "$mask_type" = "all" ]; then
        MASK_TYPE_DIR="combined"
    else
        MASK_TYPE_DIR="$mask_type"
    fi

    MAPS_DIR="saliency/saliency_maps_${SOFT_MASKING_DIR}_${MASK_TYPE_DIR}"
    OUTPUT_DIR="evaluation/evaluation_RISE_${SOFT_MASKING_DIR}_${MASK_TYPE_DIR}"

    echo "Maps directory: $MAPS_DIR"
    echo "Output directory: $OUTPUT_DIR"

    # Check if saliency maps directory exists
    if [ ! -d "$MAPS_DIR" ]; then
        echo "ERROR: Saliency maps directory $MAPS_DIR does not exist!"
        echo "Skipping this configuration..."
        return 1
    fi

    # Check if already completed
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        echo "Output directory $OUTPUT_DIR already exists and is not empty. Skipping..."
        return 0
    fi

    # Execute evaluation
    echo "Running evaluation..."
    srun python evaluate_insertion_deletion.py \
        --images ESC50_spectrograms \
        --maps_dir "$MAPS_DIR" \
        --suffix _saliency.npy \
        --output_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully"
        echo "Results saved to: $OUTPUT_DIR"
    else
        echo "✗ Evaluation failed!"
        return 1
    fi

    echo ""
}

# Batch 4 configurations:
run_evaluation "mel" "bilinear"
run_evaluation "all" "none"
run_evaluation "time" "none"

echo "=== Batch 4 Completed ==="
