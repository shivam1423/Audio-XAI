#!/usr/bin/env python
# coding: utf-8

"""
Example script for using HTSAT with RISE audio framework.

HTSAT expects:
- Raw waveforms at 32kHz
- 10 seconds of audio (320,000 samples)
- Uses internal mel spectrogram extraction (64 mel bins)
"""

import os
import sys
from src.models.htsat import HTSATModel
from src.saliency import explain_audio_unified

def main():
    # Setup paths
    checkpoint_path = "checkpoints/HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
    audio_path = "test_audio/1-137-A-32.wav"
    output_dir = "results/saliency/htsat/test"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please download or specify the correct path to HTSAT checkpoint")
        return
    
    # Check if audio exists
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        print("Please specify a valid audio file")
        return
    
    print("=" * 70)
    print("HTSAT RISE Audio Example")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load HTSAT model (automatically includes RawAudioPreprocessor)
    print("\nStep 1: Loading HTSAT model...")
    model = HTSATModel(weights_path=checkpoint_path)
    
    print(f"✓ Model loaded")
    print(f"  Input type: {model.input_type.value}")
    print(f"  Sample rate: {model.sample_rate} Hz")
    print(f"  Input length: {model.input_length} samples ({model.input_length / model.sample_rate:.1f}s)")
    print(f"  Preprocessor: {type(model.preprocessor).__name__}")
    
    # Generate RISE saliency map
    print("\nStep 2: Generating RISE saliency map...")
    saliency_map, target_class, metadata = explain_audio_unified(
        audio_path=audio_path,
        model=model,
        mask_dir="results/masks",  # Will be auto-created
        output_dir=output_dir,
        n_masks=6000,  # Standard RISE mask count
        gpu_batch=250,  # Adjust based on GPU memory
        occlusion="black",  # Occlusion type
        soft_masking="gaussian",  # Soft masking for smooth edges
        edge_sigma_px=1.0
    )
    
    print("\n" + "=" * 70)
    print("RISE Saliency Generation Complete!")
    print("=" * 70)
    print(f"Target class: {target_class}")
    print(f"Saliency shape: {saliency_map.shape}")
    print(f"Mean score: {metadata['mean_score']:.4f}")
    print(f"Std score: {metadata['std_score']:.4f}")
    print(f"Output saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

