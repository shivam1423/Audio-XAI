#!/bin/bash
#SBATCH --job-name=rise_time_mel
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=8-08:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Execute programs
srun python saliency_masking_strategy.py \
    --mask_type all \
    --soft_masking bilinear \
    --N 6000 \
    --time_stripe_frac 0.5 \
    --mel_band_frac 0.5 \
    --output_name "saliency_maps_bilinear_time_mel" \
    --mask_name "time_mel_bilinear_masks"

srun python evaluate_insertion_deletion.py \
        --images   ESC50_spectrograms \
        --maps_dir saliency/saliency_maps_bilinear_time_mel \
        --suffix   _saliency.npy \
        --output_dir evaluations/evaluation_RISE_bilinear_time_mel