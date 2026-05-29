#!/bin/bash
#SBATCH --job-name=rise
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
srun python evaluate_insertion_deletion.py \
        --images   ESC50_spectrograms \
        --maps_dir saliency_gradcam \
        --suffix   _gradcam.npy \
        --output_dir evaluation_gradcam \
        --method GradCam