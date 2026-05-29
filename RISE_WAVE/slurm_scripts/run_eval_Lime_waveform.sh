#!/bin/bash
#SBATCH --job-name=eval_gradcam_waveform
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --time=2-00:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Execute programs
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
        --audio ../ESC50/audio \
        --maps_dir results/saliency/saliency_Lime_waveform_10_segments \
        --suffix _lime_waveform.npy \
        --output_dir results/evaluations/evaluation_Lime_waveform_10_segments
