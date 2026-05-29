#!/bin/bash
#SBATCH --job-name=Wav2Vec
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

#srun python inference.py --checkpoint checkpoints/best_model_wav2vec2.pt --audio_file ../ESC50/audio/1-137-A-32.wav
srun python inference.py \
  --checkpoint checkpoints/best_model_wav2vec2.pt \
  --esc50_root ../ESC50 \
  --output_csv results/esc50_predictions.csv