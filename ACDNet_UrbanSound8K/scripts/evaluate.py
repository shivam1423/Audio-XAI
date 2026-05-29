#!/usr/bin/env python3
"""
Evaluation script for ACDNet on UrbanSound8K

Usage:
  python scripts/evaluate.py --npz_path ./data/urbansound8k_20k.npz --model_path ./trained_models/acdnet_us8k_best.pt
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACDNetConfig
from evaluation.evaluator import evaluate_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate ACDNet on UrbanSound8K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to preprocessed NPZ file (e.g., ./data/urbansound8k_20k.npz)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='[Deprecated] Use --npz_path instead'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    
    parser.add_argument(
        '--test_fold',
        type=int,
        default=10,
        help='Test fold number'
    )
    
    parser.add_argument(
        '--n_crops',
        type=int,
        default=10,
        help='Number of crops for testing'
    )
    
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create configuration
    config = ACDNetConfig()
    
    # Update config from arguments
    config.npz_path = args.npz_path
    config.data_dir = args.data_dir
    config.device = args.device
    config.test_fold = args.test_fold
    config.n_crops = args.n_crops
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Evaluate model
    try:
        results = evaluate_model(config, args.model_path, args.output_dir)
        
        print("\n✓ Evaluation completed successfully!")
        print(f"  Final accuracy: {results['accuracy']:.2f}%")
        print(f"  Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
