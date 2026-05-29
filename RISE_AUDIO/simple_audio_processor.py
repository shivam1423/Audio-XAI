#!/usr/bin/env python
# coding: utf-8

"""Simple script for processing a single audio file and saving all intermediate results."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchaudio
import torchvision.transforms as transforms

# Import modular components
from src.audio_io import AudioProcessor
from src.feature_extractor import process_masked_audio_list, audio_tensor_to_mel_spectrogram_image
from src.models.resnet50 import ResNetModel
from src.saliency import RISEAudioSaliency
from src.utils import (
    original_nfft, MASKS_DIR, DEFAULT_N_MASKS, DEFAULT_TIME_STRIPE_FRAC,
    DEFAULT_FREQ_BAND_FRAC, DEFAULT_RECT_PATCH_FRAC, DEFAULT_MEL_BAND_FRAC,
    DEFAULT_SOFT_MASKING, DEFAULT_EDGE_SIGMA_PX
)


def save_spectrogram(spectrogram, output_path, title="Spectrogram"):
    """Save spectrogram as image."""
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {output_path}")


def save_mel_spectrogram(mel_spec, output_path, title="Mel Spectrogram"):
    """Save mel spectrogram as image."""
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spec, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {output_path}")


def save_saliency_visualization(saliency_map, original_mel, output_path):
    """Save saliency map visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original mel spectrogram
    ax1.imshow(original_mel, cmap='gray', aspect='auto')
    ax1.set_title('Original Mel Spectrogram')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mel Frequency')
    
    # Saliency map
    im = ax2.imshow(saliency_map, cmap='jet', aspect='auto')
    ax2.set_title('Saliency Map')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mel Frequency')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved saliency visualization to {output_path}")


def process_single_audio_file(audio_path, output_base_dir="temp/results/saliency"):
    """Process a single audio file and save all intermediate results."""
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "spectrograms"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "masked_spectrograms"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "mel_spectrograms"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "saliency_maps"), exist_ok=True)
    
    print(f"Processing audio file: {audio_path}")
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Load original audio
    audio, sr = audio_processor.load_audio(audio_path)
    print(f"Loaded audio with sample rate: {sr}")
    
    # Generate original spectrogram
    stft = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=512,
        window_fn=torch.hann_window,
        onesided=True,
        power=2.0
    )
    
    original_spec = stft(audio)
    original_spec_db = torchaudio.transforms.AmplitudeToDB()(original_spec)
    
    # Save original spectrogram
    spec_path = os.path.join(output_base_dir, "spectrograms", f"{base_name}_original_spectrogram.png")
    save_spectrogram(original_spec_db.squeeze().numpy(), spec_path, "Original Spectrogram")
    
    # Generate original mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    
    original_mel = mel_transform(audio)
    original_mel_db = torchaudio.transforms.AmplitudeToDB()(original_mel)
    
    # Save original mel spectrogram
    mel_path = os.path.join(output_base_dir, "mel_spectrograms", f"{base_name}_original_mel.png")
    save_mel_spectrogram(original_mel_db.squeeze().numpy(), mel_path, "Original Mel Spectrogram")
    
    # Generate masked audio versions
    print("Generating masked audio versions...")
    masked_audio_list, sample_rate, mask_shape = audio_processor.generate_all_masked_audio(
        audio_path, 
        n_masks=DEFAULT_N_MASKS
    )
    print(f"Generated {len(masked_audio_list)} masked versions")
    
    # Save a few sample masked spectrograms (first 5)
    print("Saving sample masked spectrograms...")
    for i in range(min(5, len(masked_audio_list))):
        masked_spec = stft(masked_audio_list[i])
        masked_spec_db = torchaudio.transforms.AmplitudeToDB()(masked_spec)
        
        masked_spec_path = os.path.join(
            output_base_dir, "masked_spectrograms", 
            f"{base_name}_masked_{i:03d}_spectrogram.png"
        )
        save_spectrogram(
            masked_spec_db.squeeze().numpy(), 
            masked_spec_path, 
            f"Masked Spectrogram {i+1}"
        )
    
    # Convert masked audio to mel spectrogram images
    print("Converting to mel spectrogram images...")
    mel_images = process_masked_audio_list(
        masked_audio_list,
        sr=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    print(f"Generated {len(mel_images)} mel spectrogram images")
    
    # Save a few sample mel spectrograms (first 5)
    print("Saving sample mel spectrograms...")
    for i in range(min(5, len(mel_images))):
        mel_img_path = os.path.join(
            output_base_dir, "mel_spectrograms", 
            f"{base_name}_masked_{i:03d}_mel.png"
        )
        # Convert PIL image to numpy for saving
        mel_array = np.array(mel_images[i])
        save_mel_spectrogram(mel_array, mel_img_path, f"Masked Mel Spectrogram {i+1}")
    
    # Load model
    print("Loading model...")
    model = ResNetModel()
    
    # Initialize saliency generator
    print("Initializing saliency generator...")
    saliency_gen = RISEAudioSaliency(
        model=model,
        input_size=(224, 224),
        gpu_batch=250,
        soft_masking=DEFAULT_SOFT_MASKING,
        edge_sigma_px=DEFAULT_EDGE_SIGMA_PX,
        occlusion="black"
    )
    
    # Load masks
    h, w = mask_shape
    mask_path = os.path.join(MASKS_DIR, f"masks_{DEFAULT_N_MASKS}_{h}_{w}_{DEFAULT_SOFT_MASKING}_all.npy")
    print(f"Loading masks from: {mask_path}")
    saliency_gen.load_masks(mask_path=mask_path)
    
    # Generate saliency map
    print("Generating saliency map...")
    saliency_map, target_class, metadata = saliency_gen.generate_saliency_from_mel_images(
        mel_images=mel_images
    )
    
    print(f"Generated saliency map with shape: {saliency_map.shape}")
    print(f"Target class: {target_class}")
    print(f"Mean score: {metadata['mean_score']:.4f}")
    print(f"Std score: {metadata['std_score']:.4f}")
    
    # Save saliency map as numpy array
    saliency_npy_path = os.path.join(output_base_dir, "saliency_maps", f"{base_name}_saliency.npy")
    np.save(saliency_npy_path, saliency_map)
    print(f"Saved saliency map to {saliency_npy_path}")
    
    # Save saliency visualization
    saliency_viz_path = os.path.join(output_base_dir, "saliency_maps", f"{base_name}_saliency_visualization.png")
    save_saliency_visualization(saliency_map, original_mel_db.squeeze().numpy(), saliency_viz_path)
    
    # Save metadata
    metadata_path = os.path.join(output_base_dir, "saliency_maps", f"{base_name}_metadata.npz")
    np.savez(metadata_path, **metadata)
    print(f"Saved metadata to {metadata_path}")
    
    print(f"\nCompleted processing {audio_path}")
    print(f"All results saved to {output_base_dir}")
    
    return saliency_map, target_class, metadata


def main():
    """Main function to process a single audio file."""
    
    # Set the audio file path - change this to your desired audio file
    audio_file = "test_audio_22050/1-1791-A-26_22050.wav"  # You can change this path
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please update the audio_file path in the main() function")
        return
    
    # Process the audio file
    try:
        saliency_map, target_class, metadata = process_single_audio_file(audio_file)
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

