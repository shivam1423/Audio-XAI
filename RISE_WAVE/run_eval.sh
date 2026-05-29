#!/bin/bash
#SBATCH --job-name=eval_wave_test
#SBATCH --account=ifn
#SBATCH --partition=ifn
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1080:2


# RISE Waveform Evaluation - Test Run (single ESC-50 file)
# Load modules
#module load anaconda/3-5.0.1
#module load cuda/11.1

# Extra output
nvidia-smi
echo "Node: $(hostname)"
echo "======================================"

TEST_AUDIO="test_audio"

# Evaluate RISE (wav2vec2)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model wav2vec2 \
    --method rise \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/rise_wav2vec2 \
    --output_dir results/test/eval_rise_wav2vec2

# Evaluate RISE (acdnet)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model acdnet \
    --method rise \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/rise_acdnet \
    --output_dir results/test/eval_rise_acdnet

# Evaluate RISE mask-occlusion framework (wav2vec2)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model wav2vec2 \
    --method rise_mo \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/rise_mo_wav2vec2 \
    --output_dir results/test/eval_rise_mo_wav2vec2

# Evaluate RISE mask-occlusion framework (acdnet)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model acdnet \
    --method rise_mo \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/rise_mo_acdnet \
    --output_dir results/test/eval_rise_mo_acdnet

# Evaluate LIME (wav2vec2)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model wav2vec2 \
    --method lime \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/lime_wav2vec2 \
    --output_dir results/test/eval_lime_wav2vec2

# Evaluate LIME (acdnet)
srun python src/evaluation_scripts/evaluate_insertion_deletion_waveform.py \
    --dataset esc50 \
    --model acdnet \
    --method lime \
    --audio "$TEST_AUDIO" \
    --maps_dir results/test/lime_acdnet \
    --output_dir results/test/eval_lime_acdnet
