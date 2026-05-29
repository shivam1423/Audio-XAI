#!/usr/bin/env python3
"""
Finetune ACDNet on UrbanSound8K using ESC-50 pretrained checkpoint
Transfer learning for faster convergence and potentially better accuracy
"""

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACDNetConfig
from training.trainer import ACDNetTrainer


def load_pretrained_weights(model, pretrained_path, device='cuda'):
    """
    Load pretrained weights from ESC-50 checkpoint
    
    Args:
        model: ACDNet model with 10 classes (UrbanSound8K)
        pretrained_path: Path to ESC-50 checkpoint (50 classes)
        device: Device to load checkpoint to
    
    Returns:
        model: Model with pretrained weights loaded
    """
    print(f"\nLoading pretrained checkpoint from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Extract state dict (ESC-50 checkpoint uses 'weight' key)
    if 'weight' in checkpoint:
        pretrained_dict = checkpoint['weight']
        print("  Loaded from 'weight' key (ESC-50 format)")
    elif 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
        print("  Loaded from 'model_state_dict' key")
    else:
        pretrained_dict = checkpoint
        print("  Loaded directly from checkpoint")
    
    # Get current model state dict
    model_dict = model.state_dict()
    
    # Filter out layers that don't match (final classifier layer)
    # The output layer has different number of classes (50 vs 10)
    pretrained_dict_filtered = {}
    skipped_keys = []
    loaded_keys = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            # Check if shapes match
            if model_dict[k].shape == v.shape:
                pretrained_dict_filtered[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
        else:
            skipped_keys.append(f"{k} (not in model)")
    
    print(f"\n  Loaded {len(loaded_keys)} layers from pretrained checkpoint")
    print(f"  Skipped {len(skipped_keys)} layers:")
    for key in skipped_keys:
        print(f"    - {key}")
    
    # Update model weights
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)
    
    print("\n  ✓ Pretrained weights loaded successfully!")
    print(f"  ✓ Feature extractor initialized from ESC-50 (90.50% accuracy)")
    print(f"  ✓ Final classifier randomly initialized for UrbanSound8K (10 classes)")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Finetune ACDNet on UrbanSound8K using ESC-50 pretrained checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transfer Learning Strategy:
    1. Load ESC-50 pretrained checkpoint (90.50% accuracy, 50 classes)
    2. Initialize UrbanSound8K model (10 classes)
    3. Transfer all layers except final classifier
    4. Finetune with lower learning rate and fewer epochs
    
Expected Benefits:
    - Faster convergence (100-200 epochs vs 500)
    - Better feature representations (learned from 50 diverse classes)
    - Potentially higher accuracy (85-88% vs 82-85%)
    - Training time: 3-5 hours vs 15-20 hours

Example usage:
    python scripts/finetune.py \\
        --pretrained_checkpoint ../ACDNet/acdnet_weight_pruned_trained_fold4_90.50.pt \\
        --npz_path ./data/urbansound8k_20k.npz \\
        --output_dir ./finetune_models \\
        --epochs 200 \\
        --lr 0.01
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--pretrained_checkpoint',
        type=str,
        required=True,
        help='Path to ESC-50 pretrained checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to preprocessed UrbanSound8K NPZ file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./finetune_models',
        help='Directory to save finetuned models (default: ./finetune_models)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of epochs for finetuning (default: 200, less than from-scratch 500)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Initial learning rate for finetuning (default: 0.01, 10x smaller than from-scratch)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--freeze_sfeb',
        action='store_true',
        help='Freeze SFEB (spatial feature extractor) layers during finetuning'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ACDNet Transfer Learning: ESC-50 → UrbanSound8K")
    print("=" * 70)
    print(f"Pretrained Checkpoint: {args.pretrained_checkpoint}")
    print(f"NPZ Path: {args.npz_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Freeze SFEB: {args.freeze_sfeb}")
    print(f"Random Seed: {args.seed}")
    print("=" * 70)
    print()
    
    # Create configuration for UrbanSound8K
    config = ACDNetConfig()
    config.npz_path = args.npz_path
    config.trained_models_dir = args.output_dir
    config.device = args.device
    config.batch_size = args.batch_size
    config.n_epochs = args.epochs
    config.lr = args.lr
    config.random_seed = args.seed
    
    # Adjust learning rate schedule for finetuning
    # Use same relative schedule but over fewer epochs
    config.schedule = [0.3, 0.6, 0.9]  # Decay at 30%, 60%, 90% of epochs
    config.warmup = 10  # Keep warmup
    
    print("Random seed set to:", args.seed)
    print("Creating finetuning trainer\n")
    
    # Create trainer
    trainer = ACDNetTrainer(config)
    
    # Load pretrained weights
    trainer.model = load_pretrained_weights(
        trainer.model,
        args.pretrained_checkpoint,
        device=config.device
    )
    
    # Optionally freeze SFEB layers
    if args.freeze_sfeb:
        print("\nFreezing SFEB (Spatial Feature Extractor) layers...")
        frozen_params = 0
        trainable_params = 0
        
        for name, param in trainer.model.named_parameters():
            if 'sfeb' in name:
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"  Frozen: {name}")
            else:
                trainable_params += param.numel()
        
        print(f"\n  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Training only TFEB and output layers (faster finetuning)\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start finetuning
    print("=" * 70)
    print("Starting Transfer Learning (Finetuning)")
    print("=" * 70)
    print()
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Finetuning Complete!")
    print("=" * 70)
    print(f"Best model saved to: {args.output_dir}/acdnet_us8k_best.pt")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
