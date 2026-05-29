#!/bin/bash
# evaluate_saliency.sh - Evaluation script
#SBATCH --partition=gpub
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=10000M
#SBATCH --time=2-00:00:00

# Get parameters
MASK_TYPE=$1
SOFT_MASKING=$2

echo "============================================="
echo "Starting RISE evaluation"
echo "Mask Type: $MASK_TYPE"
echo "Soft Masking: $SOFT_MASKING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "============================================="

# Load modules (uncomment if needed)
#module load anaconda/3-5.0.1
#module load cuda/11.1
if [ "$SOFT_MASKING" = "none" ]; then
    SOFT_MASKING_DIR="discrete"
else
    SOFT_MASKING_DIR="$SOFT_MASKING"
fi

if [ "$MASK_TYPE" = "all" ]; then
    MASK_TYPE_DIR="combined"
else
    MASK_TYPE_DIR="$MASK_TYPE"
fi

MAPS_DIR="saliency/saliency_maps_${SOFT_MASKING_DIR}_${MASK_TYPE_DIR}"
OUTPUT_DIR="evaluation/evaluation_RISE_${SOFT_MASKING_DIR}_${MASK_TYPE_DIR}"

echo "Maps directory: $MAPS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Check if saliency maps directory exists
if [ ! -d "$MAPS_DIR" ]; then
    echo "ERROR: Saliency maps directory $MAPS_DIR does not exist!"
    exit 1
fi

# Execute evaluation
echo "Running evaluation..."
srun python evaluate_insertion_deletion.py \
    --images ESC50_spectrograms \
    --maps_dir "$MAPS_DIR" \
    --suffix _saliency.npy \
    --output_dir "$OUTPUT_DIR"

echo "Evaluation completed for mask_type=$MASK_TYPE, soft_masking=$SOFT_MASKING"
echo "Results saved to: $OUTPUT_DIR"