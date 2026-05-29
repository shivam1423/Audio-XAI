#!/usr/bin/env python3
"""
Preprocess UrbanSound8K dataset and save to NPZ format for fast loading
Based on ACDNet's original preprocessing for ESC-50

This script:
1. Loads all audio files from UrbanSound8K
2. Resamples to 20kHz
3. Saves to NPZ format organized by folds
4. Enables instant loading during training (no 30-60 min initialization)

Output NPZ Structure:
{
    'fold1': {'sounds': [audio_array, ...], 'labels': [0, 3, 7, ...]},
    'fold2': {'sounds': [...], 'labels': [...]},
    ...
    'fold10': {'sounds': [...], 'labels': [...]}
}

Usage:
    python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data

Author: ACDNet UrbanSound8K Implementation
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_directory_structure(data_dir):
    """
    Verify UrbanSound8K directory structure
    
    Expected structure:
        data_dir/
            metadata/UrbanSound8K.csv
            audio/fold1/
            audio/fold2/
            ...
            audio/fold10/
    """
    metadata_path = os.path.join(data_dir, 'metadata', 'UrbanSound8K.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please ensure UrbanSound8K is properly extracted."
        )
    
    # Check for audio folders
    audio_dir = os.path.join(data_dir, 'audio')
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            f"Expected structure: {data_dir}/audio/fold1/, fold2/, etc."
        )
    
    # Verify at least one fold exists
    fold1_path = os.path.join(audio_dir, 'fold1')
    if not os.path.exists(fold1_path):
        raise FileNotFoundError(
            f"Fold1 directory not found: {fold1_path}\n"
            f"Please check the directory structure."
        )
    
    return metadata_path, audio_dir


def load_and_resample_audio(audio_path, target_sr=20000):
    """
    Load audio file and resample to target sample rate
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 20000 Hz)
    
    Returns:
        Resampled audio as numpy array
    """
    try:
        # Load audio with librosa (automatically converts to mono)
        sound, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return sound
    except Exception as e:
        print(f"\nWarning: Failed to load {audio_path}")
        print(f"Error: {str(e)}")
        return None


def preprocess_urbansound8k(data_dir, output_dir, sr=20000, verbose=True):
    """
    Preprocess UrbanSound8K and save to NPZ format
    
    Args:
        data_dir: Path to UrbanSound8K dataset
        output_dir: Directory to save NPZ file
        sr: Target sample rate (default: 20000 Hz)
        verbose: Print progress information
    
    Returns:
        Path to created NPZ file
    """
    if verbose:
        print("=" * 70)
        print("UrbanSound8K Preprocessing to NPZ Format")
        print("=" * 70)
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target sample rate: {sr} Hz")
        print("=" * 70)
    
    # Verify directory structure
    metadata_path, audio_dir = verify_directory_structure(data_dir)
    
    # Load metadata
    if verbose:
        print(f"\nLoading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    
    if verbose:
        print(f"Total samples in dataset: {len(metadata)}")
        print(f"Number of folds: {metadata['fold'].nunique()}")
        print(f"Number of classes: {metadata['classID'].nunique()}")
    
    # Process each fold
    dataset = {}
    total_samples = 0
    failed_samples = 0
    
    for fold in range(1, 11):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing Fold {fold}")
            print(f"{'='*70}")
        
        fold_data = metadata[metadata['fold'] == fold]
        fold_sounds = []
        fold_labels = []
        
        # Process each audio file in the fold
        iterator = tqdm(fold_data.iterrows(), total=len(fold_data), 
                       desc=f"Fold {fold}", disable=not verbose)
        
        for _, row in iterator:
            audio_path = os.path.join(
                audio_dir,
                f"fold{row['fold']}",
                row['slice_file_name']
            )
            
            # Load and resample audio
            sound = load_and_resample_audio(audio_path, target_sr=sr)
            
            if sound is not None:
                fold_sounds.append(sound)
                fold_labels.append(int(row['classID']))
                total_samples += 1
            else:
                failed_samples += 1
        
        # Store fold data
        dataset[f'fold{fold}'] = {
            'sounds': fold_sounds,
            'labels': fold_labels
        }
        
        if verbose:
            print(f"Fold {fold} completed: {len(fold_sounds)} samples processed")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to NPZ
    output_filename = f'urbansound8k_{sr//1000}k.npz'
    output_path = os.path.join(output_dir, output_filename)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Saving preprocessed dataset...")
        print(f"{'='*70}")
        print(f"Output file: {output_path}")
    
    np.savez(output_path, **dataset)
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print("Preprocessing Complete!")
        print(f"{'='*70}")
        print(f"Total samples processed: {total_samples}")
        print(f"Failed samples: {failed_samples}")
        print(f"NPZ file saved to: {output_path}")
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"\nYou can now use this NPZ file for training:")
        print(f"  --npz_path {output_path}")
        print(f"{'='*70}")
    
    return output_path


def verify_npz_file(npz_path):
    """
    Verify the created NPZ file is valid and contains expected data
    
    Args:
        npz_path: Path to NPZ file
    """
    print(f"\nVerifying NPZ file: {npz_path}")
    print("-" * 70)
    
    try:
        dataset = np.load(npz_path, allow_pickle=True)
        
        print(f"NPZ file loaded successfully")
        print(f"Keys in NPZ: {list(dataset.keys())}")
        
        total_samples = 0
        for fold in range(1, 11):
            fold_key = f'fold{fold}'
            if fold_key in dataset:
                fold_data = dataset[fold_key].item()
                n_sounds = len(fold_data['sounds'])
                n_labels = len(fold_data['labels'])
                total_samples += n_sounds
                
                print(f"  {fold_key}: {n_sounds} sounds, {n_labels} labels")
                
                # Verify data integrity
                assert n_sounds == n_labels, f"Mismatch in {fold_key}: sounds != labels"
                
                # Check sample shape
                if n_sounds > 0:
                    sample_shape = fold_data['sounds'][0].shape
                    print(f"    Sample audio shape: {sample_shape}")
        
        print(f"\nTotal samples across all folds: {total_samples}")
        print(f"Verification passed! ✓")
        print("-" * 70)
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        raise


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Preprocess UrbanSound8K dataset to NPZ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data
  
This will create: ./data/urbansound8k_20k.npz
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to UrbanSound8K dataset (contains metadata/ and audio/ folders)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Directory to save NPZ file (default: ./data)'
    )
    
    parser.add_argument(
        '--sr',
        type=int,
        default=20000,
        help='Target sample rate in Hz (default: 20000)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify NPZ file after creation'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Preprocess dataset
    npz_path = preprocess_urbansound8k(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        verbose=not args.quiet
    )
    
    # Verify if requested
    if args.verify:
        verify_npz_file(npz_path)
    
    return npz_path


if __name__ == '__main__':
    main()
