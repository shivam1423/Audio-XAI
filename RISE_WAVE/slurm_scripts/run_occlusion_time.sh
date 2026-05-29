#!/bin/bash
#SBATCH --job-name=rise_time
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



#
#python cli.py --mask_type time --soft_masking gaussian --occlusion time
#python cli.py --mask_type time --soft_masking gaussian --occlusion freq
#
#python cli.py --mask_type time --soft_masking none --occlusion time
#python cli.py --mask_type time --soft_masking none --occlusion freq

python cli.py --mask_type time --soft_masking bilinear --occlusion time
#python cli.py --mask_type time --soft_masking bilinear --occlusion freq



#srun python evaluate_insertion_deletion.py \
#        --images   test_images/ \
#        --maps_dir saliency/saliency_maps_bilinear_time_occluded \
#        --suffix   _saliency.npy \
#        --output_dir evaluation_RISE_bilinear_time_occluded