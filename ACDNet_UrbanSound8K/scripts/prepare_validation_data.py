#!/usr/bin/env python3
"""
Prepare multi-crop validation/test data for ACDNet UrbanSound8K
Based on ACDNet/common/val_generator.py

This script generates pre-processed validation and test data with 10 evenly-spaced crops
per audio sample, following the original ACDNet methodology.

Usage:
    python scripts/prepare_validation_data.py \
        --npz_path ./data/urbansound8k_20k.npz \
        --output_dir ./val_data \
        --val_fold 9 \
        --test_fold 10

Output:
    val_data/fold9_val10crop.npz - Validation data with 10 crops per sample
    val_data/fold10_val10crop.npz - Test data with 10 crops per sample
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import padding, normalize, multi_crop


def prepare_multi_crop_data(npz_path, fold, input_length=30225, n_crops=10, num_classes=10):
    """
    Prepare multi-crop validation data for a specific fold
    
    Args:
        npz_path: Path to preprocessed NPZ file (urbansound8k_20k.npz)
        fold: Fold number to process
        input_length: Length of each crop (default: 30225 like ESC-50)
        n_crops: Number of crops per sample (default: 10)
        num_classes: Number of classes (default: 10)
    
    Returns:
        val_x: Array of shape (n_samples * n_crops, 1, input_length, 1)
        val_y: Array of shape (n_samples * n_crops, num_classes)
    """
    print(f"\nProcessing fold {fold}...")
    
    # Load data from NPZ
    dataset = np.load(npz_path, allow_pickle=True)
    fold_key = f'fold{fold}'
    
    if fold_key not in dataset:
        raise KeyError(f"Fold {fold} not found in NPZ file. Available: {list(dataset.keys())}")
    
    fold_data = dataset[fold_key].item()
    sounds = fold_data['sounds']
    labels = fold_data['labels']
    
    print(f"  Loaded {len(sounds)} samples from {fold_key}")
    
    # Preprocessing functions (following original ACDNet val_generator.py)
    preprocess_funcs = [
        padding(input_length // 2),
        normalize(32768.0),
        multi_crop(input_length, n_crops)
    ]
    
    all_crops = []
    all_labels = []
    
    print(f"  Applying preprocessing and multi-crop (10 crops per sample)...")
    for sound, label in tqdm(zip(sounds, labels), total=len(sounds)):
        # Apply preprocessing
        processed = sound.copy()
        for func in preprocess_funcs:
            processed = func(processed)
        
        # processed is now (n_crops, input_length)
        # Create one-hot labels repeated n_crops times
        label_one_hot = np.zeros((n_crops, num_classes), dtype=np.float32)
        label_one_hot[:, label] = 1.0
        
        all_crops.append(processed)
        all_labels.append(label_one_hot)
    
    # Stack all samples
    all_crops = np.array(all_crops)  # (n_samples, n_crops, input_length)
    all_labels = np.array(all_labels)  # (n_samples, n_crops, num_classes)
    
    # Reshape to flatten samples and crops
    # From: (n_samples, n_crops, input_length)
    # To: (n_samples * n_crops, input_length)
    val_x = all_crops.reshape(all_crops.shape[0] * all_crops.shape[1], all_crops.shape[2])
    val_y = all_labels.reshape(all_labels.shape[0] * all_labels.shape[1], all_labels.shape[2])
    
    print(f"  Final shapes: val_x={val_x.shape}, val_y={val_y.shape}")
    print(f"  Total crops: {len(val_x)} ({len(sounds)} samples × {n_crops} crops)")
    
    # Expand dimensions to match model input format: (batch, 1, input_length, 1)
    val_x = np.expand_dims(val_x, axis=1)  # (n, 1, input_length)
    val_x = np.expand_dims(val_x, axis=3)  # (n, 1, input_length, 1)
    
    print(f"  After expand_dims: val_x={val_x.shape}, val_y={val_y.shape}")
    
    return val_x, val_y


def main():
    parser = argparse.ArgumentParser(
        description='Prepare multi-crop validation data for ACDNet UrbanSound8K',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python scripts/prepare_validation_data.py \\
        --npz_path ./data/urbansound8k_20k.npz \\
        --output_dir ./val_data \\
        --val_fold 9 \\
        --test_fold 10

This will create:
    ./val_data/fold9_val10crop.npz
    ./val_data/fold10_val10crop.npz
        """
    )
    
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to preprocessed NPZ file (urbansound8k_20k.npz)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./val_data',
        help='Directory to save multi-crop validation data (default: ./val_data)'
    )
    
    parser.add_argument(
        '--val_fold',
        type=int,
        default=9,
        help='Validation fold number (default: 9)'
    )
    
    parser.add_argument(
        '--test_fold',
        type=int,
        default=10,
        help='Test fold number (default: 10)'
    )
    
    parser.add_argument(
        '--input_length',
        type=int,
        default=30225,
        help='Input length for crops (default: 30225, matching ESC-50)'
    )
    
    parser.add_argument(
        '--n_crops',
        type=int,
        default=10,
        help='Number of crops per sample (default: 10)'
    )
    
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of classes (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ACDNet UrbanSound8K - Multi-Crop Validation Data Preparation")
    print("=" * 70)
    print(f"NPZ path: {args.npz_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation fold: {args.val_fold}")
    print(f"Test fold: {args.test_fold}")
    print(f"Input length: {args.input_length}")
    print(f"Number of crops: {args.n_crops}")
    print("=" * 70)
    
    # Verify NPZ file exists
    if not os.path.exists(args.npz_path):
        print(f"\nERROR: NPZ file not found: {args.npz_path}")
        print("Please run: python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process validation fold
    print(f"\n{'='*70}")
    print(f"Processing Validation Fold {args.val_fold}")
    print(f"{'='*70}")
    val_x, val_y = prepare_multi_crop_data(
        args.npz_path,
        args.val_fold,
        args.input_length,
        args.n_crops,
        args.num_classes
    )
    
    val_output = os.path.join(args.output_dir, f'fold{args.val_fold}_val10crop.npz')
    print(f"\nSaving to: {val_output}")
    np.savez_compressed(val_output, x=val_x, y=val_y)
    
    file_size_mb = os.path.getsize(val_output) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  ✓ Validation data saved")
    
    # Process test fold
    print(f"\n{'='*70}")
    print(f"Processing Test Fold {args.test_fold}")
    print(f"{'='*70}")
    test_x, test_y = prepare_multi_crop_data(
        args.npz_path,
        args.test_fold,
        args.input_length,
        args.n_crops,
        args.num_classes
    )
    
    test_output = os.path.join(args.output_dir, f'fold{args.test_fold}_val10crop.npz')
    print(f"\nSaving to: {test_output}")
    np.savez_compressed(test_output, x=test_x, y=test_y)
    
    file_size_mb = os.path.getsize(test_output) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  ✓ Test data saved")
    
    # Summary
    print(f"\n{'='*70}")
    print("Multi-Crop Data Preparation Complete!")
    print(f"{'='*70}")
    print(f"Validation fold {args.val_fold}: {len(val_x)} crops ({len(val_x)//args.n_crops} samples)")
    print(f"Test fold {args.test_fold}: {len(test_x)} crops ({len(test_x)//args.n_crops} samples)")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - fold{args.val_fold}_val10crop.npz")
    print(f"  - fold{args.test_fold}_val10crop.npz")
    print(f"\nThese files will be used during training for validation and final testing.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
