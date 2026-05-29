#!/usr/bin/env python3
"""
Training script for ACDNet on UrbanSound8K

Usage:
  1. First, preprocess the dataset:
     python scripts/prepare_urbansound8k.py --data_dir ../UrbanSound8K --output_dir ./data
  
  2. Then train the model:
     python scripts/train.py --npz_path ./data/urbansound8k_20k.npz --output_dir ./trained_models

Note: NPZ preprocessing enables instant training start (no 30-60 min initialization wait)
"""

import argparse
import sys
import os
import random
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACDNetConfig
from training.trainer import ACDNetTrainer


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ACDNet on UrbanSound8K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to preprocessed NPZ file (e.g., ./data/urbansound8k_20k.npz). '
             'Create using: python scripts/prepare_urbansound8k.py'
    )
    
    # Optional arguments (kept for backward compatibility)
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='[Deprecated] Use --npz_path instead'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./trained_models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: 32 from config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 120 from config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Initial learning rate (default: 0.1 from config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no_bc_learning',
        action='store_true',
        help='Disable BC (Between-Class) Learning'
    )
    
    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Create configuration
    config = ACDNetConfig()
    
    # Update config from arguments
    config.update_from_args(args)
    
    # Disable BC Learning if requested
    if args.no_bc_learning:
        config.use_bc_learning = False
        print("BC Learning disabled")
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    print("creating trainer")
    # Create trainer
    trainer = ACDNetTrainer(config)
    
    # Train model
    try:
        train_history, val_history = trainer.train()
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
