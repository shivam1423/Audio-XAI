#!/bin/bash
# master_evaluation.sh - Master script to submit 5 evaluation jobs

echo "=== Submitting 5 Evaluation Batch Jobs ==="
echo "Each job will run 3 evaluations sequentially"

# Create logs directory
mkdir -p logs

# Function to create evaluation script content
create_eval_script() {
    local batch_num=$1
    shift
    local configs=("$@")

    cat << 'EOF' > eval_batch_${batch_num}.sh
#!/bin/bash
#SBATCH --job-name=eval_batch_${BATCH_NUM}
#SBATCH --partition=gpub
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=10000M
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/eval_batch_${BATCH_NUM}_%j.out
#SBATCH --error=logs/eval_batch_${BATCH_NUM}_%j.err

# Load modules (uncomment if needed)
#module load anaconda/3-5.0.1
#module load cuda/11.1

echo "=== Starting Evaluation Batch ${BATCH_NUM} ==="
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
    OUTPUT_DIR="evaluations/evaluation_RISE_${SOFT_MASKING_DIR}_${MASK_TYPE_DIR}"

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

EOF

    # Add the specific configurations for this batch
    echo "# Batch ${batch_num} configurations:" >> eval_batch_${batch_num}.sh
    for config in "${configs[@]}"; do
        IFS=',' read -r mask_type soft_masking <<< "$config"
        echo "run_evaluation \"$mask_type\" \"$soft_masking\"" >> eval_batch_${batch_num}.sh
    done

    echo "" >> eval_batch_${batch_num}.sh
    echo "echo \"=== Batch ${batch_num} Completed ===\""  >> eval_batch_${batch_num}.sh

    # Replace BATCH_NUM placeholder
    sed -i "s/\${BATCH_NUM}/${batch_num}/g" eval_batch_${batch_num}.sh
}

# Define configurations for each batch (mask_type,soft_masking)
declare -a BATCH1=("all,gaussian" "time,gaussian" "freq,gaussian")
declare -a BATCH2=("rect,gaussian" "mel,gaussian" "all,bilinear")
declare -a BATCH3=("time,bilinear" "freq,bilinear" "rect,bilinear")
declare -a BATCH4=("mel,bilinear" "all,none" "time,none")
declare -a BATCH5=("freq,none" "rect,none" "mel,none")

# Create batch scripts
echo "Creating batch scripts..."
create_eval_script 1 "${BATCH1[@]}"
create_eval_script 2 "${BATCH2[@]}"
create_eval_script 3 "${BATCH3[@]}"
create_eval_script 4 "${BATCH4[@]}"
create_eval_script 5 "${BATCH5[@]}"

# Make scripts executable
chmod +x eval_batch_*.sh

# Submit all batch jobs
echo ""
echo "Submitting batch jobs..."

JOB1=$(sbatch --parsable eval_batch_1.sh)
echo "Submitted Batch 1 (gaussian: all,time,freq): Job ID $JOB1"

JOB2=$(sbatch --parsable eval_batch_2.sh)
echo "Submitted Batch 2 (gaussian: rect,mel + bilinear: all): Job ID $JOB2"

JOB3=$(sbatch --parsable eval_batch_3.sh)
echo "Submitted Batch 3 (bilinear: time,freq,rect): Job ID $JOB3"

JOB4=$(sbatch --parsable eval_batch_4.sh)
echo "Submitted Batch 4 (bilinear: mel + none: all,time): Job ID $JOB4"

JOB5=$(sbatch --parsable eval_batch_5.sh)
echo "Submitted Batch 5 (none: freq,rect,mel): Job ID $JOB5"

echo ""
echo "=== All Jobs Submitted ==="
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB1,$JOB2,$JOB3,$JOB4,$JOB5"
echo ""
echo "Check logs in: logs/"
echo ""
echo "Batch Distribution:"
echo "  Batch 1: gaussian (all, time, freq)"
echo "  Batch 2: gaussian (rect, mel) + bilinear (all)"
echo "  Batch 3: bilinear (time, freq, rect)"
echo "  Batch 4: bilinear (mel) + none (all, time)"
echo "  Batch 5: none (freq, rect, mel)"

# Cleanup function
cat << 'EOF' > cleanup_batch_scripts.sh
#!/bin/bash
echo "Cleaning up batch scripts..."
rm -f eval_batch_*.sh
echo "Batch scripts removed."
EOF

chmod +x cleanup_batch_scripts.sh

echo ""
echo "To clean up generated scripts later, run: ./cleanup_batch_scripts.sh"