#!/bin/bash
#SBATCH --job-name=rise_all
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

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_combined_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_combined_occlusion_time

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_combined_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_combined_occlusion_freq

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_combined_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_combined_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_combined_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_combined_occlusion_freq
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_combined_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_combined_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_combined_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_combined_occlusion_freq



#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_freq_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_freq_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_freq_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_freq_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_freq_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_freq_occlusion_time

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_freq_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_freq_occlusion_freq

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_freq_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_freq_occlusion_freq

#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_freq_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_freq_occlusion_freq



#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_mel_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_mel_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_mel_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_mel_occlusion_freq
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_mel_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_mel_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_mel_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_mel_occlusion_freq
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_mel_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_mel_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_mel_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_mel_occlusion_freq
#


#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_rect_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_rect_occlusion_time
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_rect_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_rect_occlusion_freq
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_rect_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_rect_occlusion_time
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_rect_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_rect_occlusion_freq
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_rect_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_rect_occlusion_time
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_rect_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_rect_occlusion_freq


#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_time_occlusion_time \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_time_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_gaussian_time_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_gaussian_time_occlusion_freq

srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
        --images   ESC50_spectrograms/ \
        --maps_dir results/saliency/saliency_maps_discrete_time_occlusion_time \
        --suffix   _saliency.npy \
        --output_dir results/evaluations/evaluation_RISE_discrete_time_occlusion_time
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_discrete_time_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_discrete_time_occlusion_freq
#
#srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
#        --images   ESC50_spectrograms/ \
#        --maps_dir results/saliency/saliency_maps_bilinear_time_occlusion_freq \
#        --suffix   _saliency.npy \
#        --output_dir results/evaluations/evaluation_RISE_bilinear_time_occlusion_freq
#
srun python -m src.evaluation_scripts.evaluate_insertion_deletion  \
        --images   ESC50_spectrograms/ \
        --maps_dir results/saliency/saliency_maps_bilinear_time_occlusion_time \
        --suffix   _saliency.npy \
        --output_dir results/evaluations/evaluation_RISE_bilinear_time_occlusion_time