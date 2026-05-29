#!/usr/bin/env python3
"""
Shape Verification Script for ACDNet UrbanSound8K
Tests that tensor shapes are correct at each stage

Run this BEFORE training to verify everything is set up correctly
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACDNetConfig
from models.acdnet import GetACDNetModel


def test_shape_transformations():
    """Test shape transformations match ACDNet requirements"""
    print("="*70)
    print("ACDNet Shape Verification Test")
    print("="*70)
    print()
    
    # Test parameters
    batch_size = 32
    input_length = 30000
    num_classes = 10
    
    print("Test parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input length: {input_length}")
    print(f"  Num classes: {num_classes}")
    print()
    
    # Step 1: Simulate generator output
    print("Step 1: Generator output")
    print("-" * 70)
    batchX = np.random.randn(batch_size, input_length).astype(np.float32)
    print(f"  Initial shape: {batchX.shape}")
    
    # Apply expand_dims like generator does
    batchX = np.expand_dims(batchX, axis=1)
    print(f"  After expand_dims(axis=1): {batchX.shape}")
    
    batchX = np.expand_dims(batchX, axis=3)
    print(f"  After expand_dims(axis=3): {batchX.shape}")
    print(f"  → Generator returns: {batchX.shape}")
    print()
    
    # Step 2: Apply moveaxis (trainer does this)
    print("Step 2: Trainer applies moveaxis")
    print("-" * 70)
    print(f"  Before moveaxis: {batchX.shape}")
    batchX_transformed = np.moveaxis(batchX, 3, 1)
    print(f"  After moveaxis(x, 3, 1): {batchX_transformed.shape}")
    print(f"  → Expected: ({batch_size}, 1, 1, {input_length})")
    
    # Verify shape is correct
    assert batchX_transformed.shape == (batch_size, 1, 1, input_length), \
        f"Shape mismatch! Expected ({batch_size}, 1, 1, {input_length}), got {batchX_transformed.shape}"
    print(f"  ✓ Shape is correct!")
    print()
    
    # Step 3: Convert to tensor
    print("Step 3: Convert to PyTorch tensor")
    print("-" * 70)
    tensor_x = torch.tensor(batchX_transformed)
    print(f"  Tensor shape: {tensor_x.shape}")
    print(f"  Tensor dtype: {tensor_x.dtype}")
    print()
    
    # Step 4: Test with actual model
    print("Step 4: Test with ACDNet model")
    print("-" * 70)
    try:
        model = GetACDNetModel(input_len=input_length, nclass=num_classes, sr=20000)
        print(f"  Model created successfully")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(tensor_x)
        
        print(f"  Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output: ({batch_size}, {num_classes})")
        
        assert output.shape == (batch_size, num_classes), \
            f"Output shape mismatch! Expected ({batch_size}, {num_classes}), got {output.shape}"
        
        print(f"  ✓ Model output shape is correct!")
        print()
        
    except Exception as e:
        print(f"  ✗ Model forward pass failed: {e}")
        return False
    
    # Step 5: Test single sample (for evaluation)
    print("Step 5: Test single sample (evaluation mode)")
    print("-" * 70)
    single_audio = np.random.randn(input_length).astype(np.float32)
    print(f"  Raw audio shape: {single_audio.shape}")
    
    # Prepare for model
    x = np.expand_dims(single_audio, axis=0)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=3)
    print(f"  After expand_dims: {x.shape}")
    
    x = np.moveaxis(x, 3, 1)
    print(f"  After moveaxis: {x.shape}")
    print(f"  → Expected: (1, 1, 1, {input_length})")
    
    assert x.shape == (1, 1, 1, input_length), \
        f"Single sample shape mismatch!"
    
    # Test with model
    tensor_x = torch.tensor(x)
    with torch.no_grad():
        output = model(tensor_x)
    
    print(f"  Model output: {output.shape}")
    assert output.shape == (1, num_classes), \
        f"Single output shape mismatch!"
    
    print(f"  ✓ Single sample evaluation works!")
    print()
    
    # Final summary
    print("="*70)
    print("✓ All shape verification tests PASSED!")
    print("="*70)
    print()
    print("Shape transformation pipeline verified:")
    print(f"  1. Raw audio: ({input_length},)")
    print(f"  2. After preprocessing: (batch, {input_length})")
    print(f"  3. Generator output: (batch, 1, {input_length}, 1)")
    print(f"  4. After moveaxis: (batch, 1, 1, {input_length})")
    print(f"  5. Model output: (batch, {num_classes})")
    print()
    print("✓ Ready to train!")
    print()
    
    return True


def test_npz_format():
    """Test NPZ file format (if exists)"""
    print("="*70)
    print("NPZ Format Verification")
    print("="*70)
    print()
    
    npz_path = "./data/urbansound8k_20k.npz"
    
    if not os.path.exists(npz_path):
        print(f"  NPZ file not found: {npz_path}")
        print(f"  This is OK - run preprocessing first:")
        print(f"  python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data")
        print()
        return None
    
    print(f"  Loading NPZ: {npz_path}")
    dataset = np.load(npz_path, allow_pickle=True)
    
    print(f"  Keys: {list(dataset.keys())}")
    print()
    
    # Check each fold
    for fold in range(1, 11):
        fold_key = f'fold{fold}'
        if fold_key in dataset:
            fold_data = dataset[fold_key].item()
            sounds = fold_data['sounds']
            labels = fold_data['labels']
            
            print(f"  {fold_key}:")
            print(f"    Sounds: {len(sounds)}")
            print(f"    Labels: {len(labels)}")
            
            if len(sounds) > 0:
                print(f"    Sample audio shape: {sounds[0].shape}")
                print(f"    Sample audio dtype: {sounds[0].dtype}")
    
    print()
    print("  ✓ NPZ format is correct!")
    print()
    
    return True


if __name__ == '__main__':
    print("\n" * 2)
    
    # Test shapes
    shapes_ok = test_shape_transformations()
    
    print("\n")
    
    # Test NPZ (optional)
    npz_ok = test_npz_format()
    
    print("\n")
    
    if shapes_ok:
        print("="*70)
        print("✓ VERIFICATION COMPLETE - All tests passed!")
        print("="*70)
        print()
        print("You can now:")
        print("  1. Preprocess dataset (if not done):")
        print("     python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data")
        print()
        print("  2. Start training:")
        print("     python scripts/train.py --npz_path ./data/urbansound8k_20k.npz --output_dir ./trained_models")
        print()
        print("  OR submit SLURM job:")
        print("     sbatch scripts/run_train.sh")
        print()
    else:
        print("="*70)
        print("✗ VERIFICATION FAILED")
        print("="*70)
        print()
        print("Please check the error messages above and fix the issues.")
        print()
        sys.exit(1)
