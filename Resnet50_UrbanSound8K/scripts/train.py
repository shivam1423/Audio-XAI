#!/usr/bin/env python3
"""
Training script for ResNet50 on UrbanSound8K

Usage:
    python scripts/train.py --data_dir ../UrbanSound8K --output_dir ./trained_models
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ResNet50Config
from models.resnet50 import create_model, print_model_summary
from data.dataset import create_data_loaders
from training.trainer import Trainer
from utils.helpers import (
    set_seed, get_device, print_system_info, 
    check_dataset_structure, save_config, create_output_dirs
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ResNet50 on UrbanSound8K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to UrbanSound8K dataset directory'
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
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        choices=['sgd', 'adam'],
        help='Optimizer type'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='Do not use ImageNet pretrained weights'
    )
    
    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Print system info
    print_system_info()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check dataset structure
    if not check_dataset_structure(args.data_dir):
        print("\n✗ Dataset structure validation failed!")
        print("Please ensure UrbanSound8K dataset is properly organized.")
        sys.exit(1)
    
    # Create configuration
    config = ResNet50Config()
    
    # Update config from arguments
    config.update_from_args(args)
    
    # Override pretrained if specified
    if args.no_pretrained:
        config.pretrained = False
    
    # Validate configuration
    try:
        config.validate()
    except AssertionError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    
    # Display configuration
    config.display()
    
    # Create output directories
    create_output_dirs(config)
    
    # Save configuration
    save_config(config, config.output_dir)
    
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    # Create model
    model = create_model(config)
    print_model_summary(model)
    
    print("\n" + "="*70)
    print("Initializing Trainer")
    print("="*70)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train model
    try:
        print("\nStarting training...")
        train_history, val_history = trainer.train()
        
        print("\n" + "="*70)
        print("Training Completed Successfully!")
        print("="*70)
        print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Best model saved to: {os.path.join(config.output_dir, 'best_model.pth')}")
        print(f"Training log saved to: {trainer.log_file}")
        print("\nTo evaluate the model, run:")
        print(f"python scripts/evaluate.py --data_dir {args.data_dir} "
              f"--model_path {os.path.join(config.output_dir, 'best_model.pth')}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
