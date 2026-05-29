#!/bin/bash
#SBATCH --job-name=examples
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=1000M
#SBATCH --time=02:00:00

# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

srun python acdnet_esc50_inference.py     --model_path /beegfs/work_fast/pandey/Thesis/RISE_dev/ACDNet/torch/resources/pretrained_models/acdnet_weight_pruned_trained_fold4_90.50.pt     --esc50_root /beegfs/work_fast/pandey/Thesis/RISE_dev/ACDNet/datasets/esc50/ESC-50-master     --output_csv esc50_acdnet_predictions.csv