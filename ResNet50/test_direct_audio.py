#!/usr/bin/env python
"""
Quick test to verify ResNet50 can train on direct audio processing.

This script:
1. Loads a few audio samples from ESC-50
2. Creates a mini-batch
3. Verifies the dataset and preprocessing work correctly
4. Shows sample output shapes

Run this before starting full training to catch any issues early.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.append(str(Path(__file__).parent))

from utils import ESC50SpectrogramDataset, preprocess


def test_dataset():
    """Test ESC50SpectrogramDataset with direct audio processing."""
    
    print("="*70)
    print("Testing ResNet50 Direct Audio Processing")
    print("="*70)
    
    # Find ESC-50 dataset
    esc50_root = Path(__file__).parent / "ESC50"
    if not esc50_root.exists():
        esc50_root = Path(__file__).parent.parent / "ESC50"
    
    if not esc50_root.exists():
        print(f"\n✗ Error: ESC-50 dataset not found!")
        print(f"  Expected location: {esc50_root}")
        print(f"\n  Please ensure ESC-50 is downloaded and extracted.")
        return False
    
    print(f"\n✓ Found ESC-50 dataset at: {esc50_root}")
    
    # Test dataset creation
    print("\n[1/5] Creating ESC50SpectrogramDataset...")
    try:
        dataset = ESC50SpectrogramDataset(str(esc50_root), transform=preprocess)
        print(f"  ✓ Dataset created successfully")
        print(f"  ✓ Total samples: {len(dataset)}")
        print(f"  ✓ Number of classes: {len(dataset.classes)}")
        print(f"  ✓ Sample classes: {dataset.classes[:5]}")
    except Exception as e:
        print(f"  ✗ Failed to create dataset: {e}")
        return False
    
    # Test single sample
    print("\n[2/5] Loading single sample...")
    try:
        img_tensor, label = dataset[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"  ✓ Image shape: {img_tensor.shape} (expected: 3x224x224)")
        print(f"  ✓ Label: {label} ({dataset.classes[label]})")
        print(f"  ✓ Tensor dtype: {img_tensor.dtype}")
        print(f"  ✓ Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Verify shape
        if img_tensor.shape != torch.Size([3, 224, 224]):
            print(f"  ✗ ERROR: Expected shape (3, 224, 224), got {img_tensor.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test DataLoader
    print("\n[3/5] Creating DataLoader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )
        print(f"  ✓ DataLoader created successfully")
        print(f"  ✓ Batch size: 4")
        print(f"  ✓ Total batches: {len(loader)}")
    except Exception as e:
        print(f"  ✗ Failed to create DataLoader: {e}")
        return False
    
    # Test batch loading
    print("\n[4/5] Loading a batch...")
    try:
        batch_imgs, batch_labels = next(iter(loader))
        print(f"  ✓ Batch loaded successfully")
        print(f"  ✓ Batch images shape: {batch_imgs.shape} (expected: 4x3x224x224)")
        print(f"  ✓ Batch labels shape: {batch_labels.shape} (expected: 4)")
        print(f"  ✓ Labels in batch: {batch_labels.tolist()}")
        
        # Verify batch shape
        if batch_imgs.shape != torch.Size([4, 3, 224, 224]):
            print(f"  ✗ ERROR: Expected batch shape (4, 3, 224, 224), got {batch_imgs.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test ResNet50 compatibility
    print("\n[5/5] Testing ResNet50 compatibility...")
    try:
        from torchvision import models
        import torch.nn as nn
        
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 50)  # ESC-50 has 50 classes
        model.eval()
        
        with torch.no_grad():
            output = model(batch_imgs)
        
        print(f"  ✓ Model inference successful")
        print(f"  ✓ Output shape: {output.shape} (expected: 4x50)")
        print(f"  ✓ Predicted classes: {output.argmax(dim=1).tolist()}")
        
        if output.shape != torch.Size([4, 50]):
            print(f"  ✗ ERROR: Expected output shape (4, 50), got {output.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Failed model inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print("\n" + "="*70)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("="*70)
    print("\nThe ResNet50 training pipeline is ready!")
    print("\nYou can now train the model using:")
    print("  python -m training.fine_tune_resnet --esc50_root ESC50 --epochs 50 --batch_size 32 --kfolds 5")
    print("\nOr submit to SLURM:")
    print("  sbatch run.sh")
    print()
    
    return True


def main():
    """Run all tests."""
    success = test_dataset()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
