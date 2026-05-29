#!/bin/bash
#SBATCH --job-name=rise
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
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
echo -e "Working directory: $(pwd)"

# Execute programs
srun python resnet50_esc50_per_file_inference.py \
  --esc50_root ../ESC50 \
  --checkpoints_dir .  \
  --output_dir output/ \
  --device cuda \
  --batch_size 32
