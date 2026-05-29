#!/usr/bin/env python3
"""
Evaluation script for ResNet50 on UrbanSound8K

Usage:
    python scripts/evaluate.py --data_dir ../UrbanSound8K --model_path ./trained_models/best_model.pth
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ResNet50Config
from data.dataset import create_data_loaders
from evaluation.evaluator import Evaluator, load_model_for_evaluation
from utils.helpers import set_seed, print_system_info, check_dataset_structure


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate ResNet50 on UrbanSound8K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to UrbanSound8K dataset directory'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
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
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--test_fold',
        type=int,
        default=10,
        help='Test fold number (1-10)'
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
    
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Print system info
    print_system_info()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check model file exists
    if not os.path.exists(args.model_path):
        print(f"\n✗ Model file not found: {args.model_path}")
        sys.exit(1)
    
    print(f"✓ Model file found: {args.model_path}")
    
    # Check dataset structure
    if not check_dataset_structure(args.data_dir):
        print("\n✗ Dataset structure validation failed!")
        print("Please ensure UrbanSound8K dataset is properly organized.")
        sys.exit(1)
    
    # Create configuration
    config = ResNet50Config()
    config.data_dir = args.data_dir
    config.results_dir = args.output_dir
    config.device = args.device
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.test_fold = args.test_fold
    
    # Validate configuration
    try:
        config.validate()
    except AssertionError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    
    # Display configuration
    print("\n" + "="*70)
    print("Evaluation Configuration")
    print("="*70)
    print(f"Dataset: {config.dataset}")
    print(f"Data directory: {config.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {config.results_dir}")
    print(f"Test fold: {config.test_fold}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print("="*70)
    
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    # Create data loaders (we only need test_loader)
    try:
        _, _, test_loader = create_data_loaders(config)
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✓ Test dataset loaded successfully")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    print("\n" + "="*70)
    print("Loading Model")
    print("="*70)
    
    # Load trained model
    try:
        model = load_model_for_evaluation(args.model_path, config)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Initializing Evaluator")
    print("="*70)
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, config)
    
    # Run evaluation
    try:
        results = evaluator.evaluate()
        
        # Save results
        print("\n" + "="*70)
        print("Saving Results")
        print("="*70)
        evaluator.save_results(results, config.results_dir)
        
        print("\n" + "="*70)
        print("Evaluation Completed Successfully!")
        print("="*70)
        print(f"\nOverall Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"Macro F1-Score: {results['overall']['f1_macro']:.4f}")
        print(f"\nResults saved to: {config.results_dir}")
        print(f"  - evaluation_results.json")
        print(f"  - confusion_matrix.png")
        print(f"  - per_class_metrics.png")
        print(f"  - predictions.npy")
        print(f"  - labels.npy")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n✗ Evaluation interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
