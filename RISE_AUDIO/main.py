#!/usr/bin/env python
# coding: utf-8

"""Main script for TF-Structured RISE: Orchestrates all modular components."""

# Ensure all prints show elapsed runtime from process start
from src import timed_print  # noqa: F401

import os
import torch
import glob
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
import numpy as np
from src.audio_io import AudioProcessor
from src.feature_extractor import process_masked_audio_list

from src.saliency import explain_audio_unified
import matplotlib.pyplot as plt
# Setup CUDA
cudnn.benchmark = True
# from src.utils import (
#     original_nfft, MASKS_DIR, DEFAULT_N_MASKS, DEFAULT_TIME_STRIPE_FRAC,
#     DEFAULT_FREQ_BAND_FRAC, DEFAULT_RECT_PATCH_FRAC, DEFAULT_MEL_BAND_FRAC,
#     DEFAULT_SOFT_MASKING, DEFAULT_EDGE_SIGMA_PX, DEFAULT_SINGLE_BLOCK
# )
from src.utils import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, MODEL_GPU_BATCH,
    MEL_SR, MEL_N_FFT, MEL_HOP_LENGTH, MEL_N_MELS, DEFAULT_EDGE_SIGMA_PX,
    MASKS_DIR, DEFAULT_N_MASKS,DEFAULT_SINGLE_BLOCK, DEFAULT_TIME_STRIPE_FRAC,
    DEFAULT_FREQ_BAND_FRAC, DEFAULT_RECT_PATCH_FRAC, DEFAULT_MEL_BAND_FRAC,
    DEFAULT_SOFT_MASKING, DEFAULT_OCCLUSION,
    DEFAULT_MODEL_TYPE, MODEL_PATHS, MODEL_SAMPLE_RATES
)
from src.evaluation import evaluate_audio_saliency_maps
import gc

def visualize_saliency(saliency_map, audio_path, output_dir):
    """Visualize saliency map over linear spectrogram of the original resampled audio."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Prepare audio and linear spectrogram (not log-mel), using the same processor config
    from src.audio_io import AudioProcessor
    audio_processor = AudioProcessor()
    audio, sr = audio_processor.load_audio(audio_path)  # resampled to processor.target_sr

    # Complex STFT -> magnitude spectrogram
    spec_complex = audio_processor.audio_to_spectrogram(audio)  # (1, F, T) complex
    spec_mag = torch.abs(spec_complex).squeeze(0)              # (F, T) linear magnitude

    # Convert to log scale for better visualization (like typical spectrograms)
    spec_log = torch.log(spec_mag + 1e-8)  # Add small epsilon to avoid log(0)
    spec_np = spec_log.detach().cpu().numpy()

    # Normalize for display
    spec_min, spec_max = spec_np.min(), spec_np.max()
    if spec_max > spec_min:
        spec_np = (spec_np - spec_min) / (spec_max - spec_min)
    else:
        spec_np[:] = 0.0

    # Ensure saliency matches spectrogram shape; if not, resize
    sal = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    Hs, Ws = spec_np.shape
    if sal.shape[-2:] != (Hs, Ws):
        sal = F.interpolate(sal, size=(Hs, Ws), mode='bilinear', align_corners=False)
    sal_np = sal.squeeze().detach().cpu().numpy()

    # Normalize saliency for visualization
    smin, smax = float(sal_np.min()), float(sal_np.max())
    if smax > smin:
        sal_np = (sal_np - smin) / (smax - smin)
    else:
        sal_np[:] = 0.0

    plt.figure(figsize=(12, 5))

    # Left: log spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(spec_np, cmap='gray', aspect='auto', origin='lower')  # lower freqs at bottom
    plt.title('Log Magnitudes of Original Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    # Right: overlay saliency on spectrogram
    plt.subplot(1, 2, 2)
    plt.imshow(spec_np, cmap='gray', aspect='auto', origin='lower')
    plt.imshow(sal_np, cmap='jet', aspect='auto', origin='lower')
    plt.title('Saliency over Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Saliency')

    # Save
    output_path = os.path.join(output_dir, f"saliency_vis_{os.path.basename(audio_path)}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def run_evaluation(input_dir, output_dir, model, steps=224):
    """Run evaluation on generated saliency maps."""
    saliency_dir = os.path.join(output_dir)
    eval_output_dir = os.path.join(output_dir, "evaluation")
    
    print(f"\n=== RUNNING EVALUATION ===")
    print(f"Audio directory: {input_dir}")
    print(f"Saliency directory: {saliency_dir}")
    print(f"Evaluation output: {eval_output_dir}")
    
    evaluate_audio_saliency_maps(
        audio_dir=input_dir,
        saliency_dir=saliency_dir,
        output_dir=eval_output_dir,
        model=model,
        steps=steps,
        verbose=1
    )

def main():
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    gpu_batch = MODEL_GPU_BATCH.get(DEFAULT_MODEL_TYPE)
    print(f"Using GPU batch size: {gpu_batch}")
    occlusion_type = DEFAULT_OCCLUSION

    mask_dir = MASKS_DIR
    n_masks = DEFAULT_N_MASKS

    # Create output directory with occlusion type in name
    output_dir_with_occlusion = f"{output_dir}_{occlusion_type}"
    os.makedirs(output_dir_with_occlusion, exist_ok=True)

    # ===== LOAD MODEL BASED ON CONFIGURATION =====
    print("=" * 70)
    print(f"Loading model: {DEFAULT_MODEL_TYPE.upper()}")
    print("=" * 70)
    
    if DEFAULT_MODEL_TYPE == 'resnet':
        from src.models.resnet50 import ResNetModel
        model = ResNetModel(weights_path=MODEL_PATHS['resnet'])
        print("✓ Loaded ResNet50 model (spectrogram-based)")
        
    elif DEFAULT_MODEL_TYPE == 'wav2vec2':
        from src.models.wav2vec2 import Wav2Vec2Model
        model = Wav2Vec2Model(weights_path=MODEL_PATHS['wav2vec2'])
        print("✓ Loaded Wav2Vec2 model (raw audio-based)")
        
    elif DEFAULT_MODEL_TYPE == 'htsat':
        from src.models.htsat import HTSATModel
        model = HTSATModel(weights_path=MODEL_PATHS['htsat'])
        print("✓ Loaded HTSAT model (raw audio-based, 32kHz)")

    elif DEFAULT_MODEL_TYPE == 'acdnet':
        from src.models.acdnet import ACDNetModel
        model = ACDNetModel(weights_path=MODEL_PATHS['acdnet'])
        print("✓ Loaded ACDNet model (raw audio-based)")
        
    else:
        raise ValueError(f"Unknown model type: {DEFAULT_MODEL_TYPE}")
    
    print(f"Model sample rate: {MODEL_SAMPLE_RATES.get(DEFAULT_MODEL_TYPE, 'N/A')} Hz")
    print("=" * 70)

    if os.path.exists(input_dir):
        search_pattern = os.path.join(input_dir, "*.wav")
        file_paths = glob.glob(search_pattern)
    else:
        print(f"Input directory {input_dir} not found")
        return

    print(f"\nFound {len(file_paths)} audio files")
    print(f"Using occlusion type: {occlusion_type}")
    print(f"Output directory: {output_dir_with_occlusion}\n")

    # Filter out files that already have saliency maps
    files_to_process = []
    files_skipped = []
    for input_audio_file in file_paths:
        audio_basename = os.path.basename(input_audio_file)
        saliency_path = os.path.join(output_dir_with_occlusion, f"{audio_basename}_saliency.npy")
        
        if os.path.exists(saliency_path):
            files_skipped.append(input_audio_file)
        else:
            files_to_process.append(input_audio_file)
    
    print(f"Files already processed (skipping): {len(files_skipped)}")
    print(f"Files to process: {len(files_to_process)}")
    if files_skipped:
        print(f"\nSkipped files (showing first 10):")
        for f in files_skipped[:10]:
            print(f"  - {os.path.basename(f)}")
        if len(files_skipped) > 10:
            print(f"  ... and {len(files_skipped) - 10} more")
    print()

    # Process each audio file
    for i, input_audio_file in enumerate(files_to_process):
        print(f"\n{'='*70}")
        print(f"Processing {i + 1}/{len(files_to_process)}: {os.path.basename(input_audio_file)}")
        print(f"{'='*70}")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # ===== USE UNIFIED FUNCTION =====
            saliency_map, target_class, metadata = explain_audio_unified(
                audio_path=input_audio_file,
                model=model,
                mask_dir=mask_dir,
                output_dir=output_dir_with_occlusion,
                n_masks=n_masks,
                gpu_batch=gpu_batch,
                soft_masking=DEFAULT_SOFT_MASKING,
                edge_sigma_px=DEFAULT_EDGE_SIGMA_PX,
                occlusion=occlusion_type,
                time_stripe_frac=DEFAULT_TIME_STRIPE_FRAC,
                freq_band_frac=DEFAULT_FREQ_BAND_FRAC,
                rect_patch_frac=DEFAULT_RECT_PATCH_FRAC,
                mel_band_frac=DEFAULT_MEL_BAND_FRAC,
                target_mel_sr=MEL_SR,
                single_block=DEFAULT_SINGLE_BLOCK,
            )
            # ================================

            print(f"\n{'='*70}")
            print(f"RESULTS for {os.path.basename(input_audio_file)}")
            print(f"{'='*70}")
            print(f"Saliency map shape: {saliency_map.shape}")
            print(f"Target class: {target_class}")
            print(f"Mean score: {metadata['mean_score']:.4f}")
            print(f"Std score: {metadata['std_score']:.4f}")
            print(f"Model type: {metadata.get('model_type', 'spectrogram')}")
            print(f"Occlusion type: {metadata.get('occlusion_type', 'unknown')}")
            print(f"{'='*70}")

            # Create visualization
            # saliency_map = np.load(os.path.join(output_dir, f"saliency_{os.path.basename(input_audio_file)}_saliency.npy"))
            visualize_saliency(saliency_map, input_audio_file, output_dir_with_occlusion)

            # Clear memory after each file
            del saliency_map, target_class, metadata
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n❌ Error processing {input_audio_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"✓ Completed processing {len(files_to_process)} files")
    if files_skipped:
        print(f"✓ Skipped {len(files_skipped)} already processed files")
    print(f"✓ Results saved to {output_dir_with_occlusion}")
    print(f"{'='*70}")

    # Run evaluation after all saliency maps are generated
    run_evaluation(input_dir, output_dir_with_occlusion, model)


if __name__ == "__main__":
    main()