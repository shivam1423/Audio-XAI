#!/usr/bin/env python
"""
Verification script to ensure ResNet50 training preprocessing matches 
RISE_audio evaluation preprocessing.

This script loads a sample audio file and processes it using both:
1. ResNet50 training pipeline (utils.audio_to_mel_spectrogram_image)
2. RISE_audio evaluation pipeline (ResNetPreprocessor)

It then compares the outputs to verify they are identical.
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "RISE_audio"))

import torch
import numpy as np
from PIL import Image

# Import training preprocessing
from utils import audio_to_mel_spectrogram_image, preprocess

# Import evaluation preprocessing
try:
    from src.preprocessor import ResNetPreprocessor
    RISE_AUDIO_AVAILABLE = True
except ImportError:
    print("[WARNING] RISE_audio not available. Skipping comparison.")
    RISE_AUDIO_AVAILABLE = False


def verify_preprocessing_consistency(audio_path: str):
    """
    Verify that training and evaluation preprocessing produce identical results.
    
    Args:
        audio_path: Path to an ESC-50 audio file
    """
    print(f"\n{'='*70}")
    print(f"Verifying preprocessing consistency for: {Path(audio_path).name}")
    print(f"{'='*70}\n")
    
    # 1. Training pipeline (ResNet50/utils.py)
    print("[1/3] Processing with training pipeline (utils.py)...")
    train_img = audio_to_mel_spectrogram_image(audio_path)
    train_tensor = preprocess(audio_path)  # Returns 3x224x224
    
    print(f"  ✓ Training output shape: {train_tensor.shape}")
    print(f"  ✓ Training PIL Image size: {train_img.size}")
    print(f"  ✓ Training tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    if not RISE_AUDIO_AVAILABLE:
        print("\n[2/3] Skipping RISE_audio comparison (not available)")
        print("\n[3/3] Cannot verify consistency without RISE_audio")
        return
    
    # 2. Evaluation pipeline (RISE_audio/src/preprocessor.py)
    print("\n[2/3] Processing with evaluation pipeline (RISE_audio)...")
    preprocessor = ResNetPreprocessor()
    
    # Load audio as tensor (simulating RISE_audio pipeline)
    import torchaudio
    audio_tensor, orig_sr = torchaudio.load(audio_path)
    
    eval_img = preprocessor.process_original_audio(audio_tensor, orig_sr)
    
    print(f"  ✓ Evaluation PIL Image size: {eval_img.size}")
    
    # Convert both to numpy arrays for comparison
    train_img_array = np.array(train_img)
    eval_img_array = np.array(eval_img)
    
    # 3. Compare outputs
    print("\n[3/3] Comparing outputs...")
    
    # Check shapes
    if train_img_array.shape != eval_img_array.shape:
        print(f"  ✗ SHAPE MISMATCH!")
        print(f"    Training: {train_img_array.shape}")
        print(f"    Evaluation: {eval_img_array.shape}")
        return False
    
    print(f"  ✓ Shape match: {train_img_array.shape}")
    
    # Check pixel values
    pixel_diff = np.abs(train_img_array.astype(float) - eval_img_array.astype(float))
    max_diff = pixel_diff.max()
    mean_diff = pixel_diff.mean()
    
    print(f"  ✓ Max pixel difference: {max_diff:.6f}")
    print(f"  ✓ Mean pixel difference: {mean_diff:.6f}")
    
    # Check if identical (allowing small numerical differences)
    if max_diff < 1e-5:
        print(f"\n{'='*70}")
        print("✓✓✓ VERIFICATION PASSED: Preprocessing pipelines are IDENTICAL! ✓✓✓")
        print(f"{'='*70}\n")
        return True
    elif max_diff < 1.0:
        print(f"\n{'='*70}")
        print("⚠ VERIFICATION WARNING: Small differences detected (likely numerical)")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"\n{'='*70}")
        print("✗✗✗ VERIFICATION FAILED: Significant differences detected! ✗✗✗")
        print(f"{'='*70}\n")
        
        # Show sample differences
        print("Sample pixel differences (first 5x5 region):")
        print(pixel_diff[:5, :5])
        return False


def main():
    """Run verification on sample ESC-50 files."""
    
    # Find ESC-50 dataset
    esc50_root = Path(__file__).parent / "ESC50"
    if not esc50_root.exists():
        esc50_root = Path(__file__).parent.parent / "ESC50"
    
    if not esc50_root.exists():
        print(f"Error: ESC-50 dataset not found at {esc50_root}")
        print("Please provide the path to ESC-50 directory:")
        esc50_root = input("> ").strip()
        esc50_root = Path(esc50_root)
    
    audio_dir = esc50_root / "audio"
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found at {audio_dir}")
        sys.exit(1)
    
    # Get sample files
    audio_files = sorted(audio_dir.glob("*.wav"))[:3]  # Test first 3 files
    
    if not audio_files:
        print(f"Error: No .wav files found in {audio_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(audio_files)} ESC-50 audio files")
    print(f"Testing on {len(audio_files)} sample files...\n")
    
    # Verify each file
    results = []
    for audio_path in audio_files:
        result = verify_preprocessing_consistency(str(audio_path))
        results.append((audio_path.name, result))
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    for filename, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {filename}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print(f"\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe ResNet50 training pipeline is now consistent with RISE_audio evaluation!")
        print("You can safely train using direct audio processing.\n")
        return 0
    else:
        print(f"\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the differences above and fix the preprocessing pipeline.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
