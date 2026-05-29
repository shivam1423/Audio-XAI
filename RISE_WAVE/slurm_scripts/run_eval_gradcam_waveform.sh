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
        --audio_dir test_audio/ \
        --maps_dir results/saliency/saliency_gradcam_waveform1 \
        --suffix _gradcam.npy \
        --output_dir results/evaluations/evaluation_gradcam_waveform1 \
        --checkpoint /beegfs/work_fast/pandey/Thesis/RISE_dev/Wav2Vec2/checkpoints/best_model_wav2vec2.pt \
        --step 224 \
        --kernel_size 100
