#!/bin/bash
#SBATCH --job-name=sal_RISE_wav
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=4000M
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
#srun python -u src/saliency_scripts/Saliency_RISE_waveform_mask_occlusion_framework.py \
#    --mask_type all \
#    --contiguous_frac 0.5 \
#    --scattered_frac 0.5 \
#    --soft_masking discrete \
#    --occlusion zeros \
#    --audio_dir ../ESC50/audio \
#    --output_dir results/saliency/ESC50_saliency_RISE_waveform_combined_discrete_occlusion_zeros

srun python -u src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --audio ../ESC50/audio \
    --maps_dir results/saliency/ESC50_saliency_RISE_waveform_combined_discrete_occlusion_zeros \
    --suffix _rise_waveform.npy \
    --output_dir results/evaluations/ESC50_evaluation_RISE_waveform_combined_discrete_occlusion_zeros
