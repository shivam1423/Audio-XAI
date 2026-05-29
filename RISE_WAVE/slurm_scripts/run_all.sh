#!/bin/bash
# run_all.sh - Main submission script

# Define your arrays based on your script
MASK_TYPES=("all" "time" "freq" "rect" "mel")
SOFT_MASKING=("gaussian" "bilinear" "none")

echo "Creating necessary directories..."
mkdir -p logs

echo "Submitting saliency generation jobs..."
for mask_type in "${MASK_TYPES[@]}"; do
    for soft_mask in "${SOFT_MASKING[@]}"; do
        # Submit generation job
        gen_job_id=$(sbatch --parsable \
            --job-name="rise_gen_${mask_type}_${soft_mask}" \
            --output="logs/%j_gen_${mask_type}_${soft_mask}.out" \
            --error="logs/%j_gen_${mask_type}_${soft_mask}.err" \
            generate_saliency.sh "$mask_type" "$soft_mask")

        echo "Submitted generation job $gen_job_id for mask_type=$mask_type, soft_masking=$soft_mask"

        # Submit evaluation job with dependency on generation
        eval_job_id=$(sbatch --parsable \
            --dependency=afterok:$gen_job_id \
            --job-name="rise_eval_${mask_type}_${soft_mask}" \
            --output="logs/eval_${mask_type}_${soft_mask}_%j.out" \
            --error="logs/eval_${mask_type}_${soft_mask}_%j.err" \
            evaluate_saliency.sh "$mask_type" "$soft_mask")

        echo "Submitted evaluation job $eval_job_id (depends on $gen_job_id)"
    done
done

echo "All jobs submitted! Total: 15 generation + 15 evaluation = 30 jobs"
echo ""
echo "To monitor jobs: squeue -u \$USER"
echo "To cancel all generation jobs: scancel --name='rise_gen_*'"
echo "To cancel all evaluation jobs: scancel --name='rise_eval_*'"
echo "To cancel all jobs: scancel --name='rise_*'"