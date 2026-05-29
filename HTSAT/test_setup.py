"""
Test script to verify HTSAT evaluation setup
This script checks that all components are properly configured before running full evaluation.
"""

import os
import sys
import torch
import torchaudio

print("="*60)
print("HTSAT Setup Verification")
print("="*60)

# Test 1: Check imports
print("\n1. Checking imports...")
try:
    from htsat_config import config
    from htsat_model import HTSAT, load_htsat_checkpoint
    from esc50_dataset import ESC50Dataset, get_dataloader
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check checkpoint file
print("\n2. Checking checkpoint file...")
if os.path.exists(config.pretrained_checkpoint):
    size_mb = os.path.getsize(config.pretrained_checkpoint) / (1024 * 1024)
    print(f"   ✓ Checkpoint found: {config.pretrained_checkpoint}")
    print(f"   Size: {size_mb:.2f} MB")
else:
    print(f"   ✗ Checkpoint not found: {config.pretrained_checkpoint}")
    sys.exit(1)

# Test 3: Check audio directory
print("\n3. Checking audio directory...")
if os.path.exists(config.audio_dir):
    wav_files = [f for f in os.listdir(config.audio_dir) if f.endswith('.wav')]
    print(f"   ✓ Audio directory found: {config.audio_dir}")
    print(f"   Number of .wav files: {len(wav_files)}")
    
    if len(wav_files) == 0:
        print("   ⚠ Warning: No .wav files found in audio directory")
else:
    print(f"   ✗ Audio directory not found: {config.audio_dir}")
    sys.exit(1)

# Test 4: Check PyTorch and CUDA
print("\n4. Checking PyTorch setup...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   TorchAudio version: {torchaudio.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠ Warning: CUDA not available, will use CPU (slower)")

# Test 5: Test dataset loading
print("\n5. Testing dataset loading...")
try:
    test_dataset = ESC50Dataset(
        audio_dir=config.audio_dir,
        target_folds=[config.val_fold],
        sample_rate=config.sample_rate,
        clip_samples=config.clip_samples
    )
    print(f"   ✓ Dataset loaded successfully")
    print(f"   Validation samples (fold {config.val_fold}): {len(test_dataset)}")
    
    # Test loading one sample
    if len(test_dataset) > 0:
        waveform, label, filename = test_dataset[0]
        print(f"   ✓ Sample loaded: {filename}")
        print(f"   Waveform shape: {waveform.shape}")
        print(f"   Label: {label} ({config.class_labels[label]})")
except Exception as e:
    print(f"   ✗ Dataset loading error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test model creation
print("\n6. Testing model creation...")
try:
    test_model = HTSAT(num_classes=config.num_classes)
    print(f"   ✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in test_model.parameters())
    trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Test checkpoint loading
print("\n7. Testing checkpoint loading...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_htsat_checkpoint(
        config.pretrained_checkpoint,
        num_classes=config.num_classes,
        device=device
    )
    print(f"   ✓ Checkpoint loaded successfully")
    print(f"   Model on device: {next(model.parameters()).device}")
except Exception as e:
    print(f"   ✗ Checkpoint loading error: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Test forward pass
print("\n8. Testing forward pass...")
try:
    if len(test_dataset) > 0:
        # Load a single sample
        waveform, label, filename = test_dataset[0]
        waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            output = model(waveform)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Input shape: {waveform.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Predicted class: {torch.argmax(output, dim=1).item()} ({config.class_labels[torch.argmax(output, dim=1).item()]})")
        print(f"   True class: {label} ({config.class_labels[label]})")
except Exception as e:
    print(f"   ✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "="*60)
print("Setup Verification Complete!")
print("="*60)
print("\nConfiguration Summary:")
print(f"  Validation fold: {config.val_fold}")
print(f"  Training folds: {config.train_folds}")
print(f"  Number of classes: {config.num_classes}")
print(f"  Sample rate: {config.sample_rate} Hz")
print(f"  Clip length: {config.clip_samples} samples ({config.clip_samples/config.sample_rate:.1f}s)")
print(f"  Batch size: {config.batch_size}")
print(f"  Device: {device}")
print("\nYou can now run the full evaluation with:")
print("  ./run_evaluation.sh")
print("or")
print("  python evaluate.py")
print("="*60)




