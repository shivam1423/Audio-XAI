#!/usr/bin/env python3
"""
Main training script for Wav2Vec2 UrbanSound8K fine-tuning
"""
import argparse
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.urbansound8k_dataset import create_data_loaders
from model.wav2vec2_classifier import create_model
from training.trainer import Wav2Vec2Trainer
from utils.helpers import (
    set_seed, 
    get_device, 
    create_directory_structure,
    print_model_summary
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 on UrbanSound8K")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../UrbanSound8K",
                       help="Path to UrbanSound8K dataset (relative for cluster)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base",
                       help="Pre-trained Wav2Vec2 model name")
    parser.add_argument("--freeze_feature_extractor", action="store_true",
                       help="Freeze Wav2Vec2 feature extractor")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout rate for classifier")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    
    # Data arguments
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=4.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Fold configuration
    parser.add_argument("--train_folds", type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Folds to use for training")
    parser.add_argument("--val_fold", type=int, default=9,
                       help="Fold to use for validation")
    parser.add_argument("--test_fold", type=int, default=10,
                       help="Fold to use for testing")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Log directory")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--test_only", action="store_true",
                       help="Only run testing (requires checkpoint)")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directory structure
    create_directory_structure(".")
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: UrbanSound8K dataset not found at {args.data_dir}")
        print(f"Please ensure the dataset is available at the specified path.")
        print(f"Expected structure:")
        print(f"  {args.data_dir}/")
        print(f"    ├── audio/")
        print(f"    │   ├── fold1/")
        print(f"    │   ├── fold2/")
        print(f"    │   ├── ...")
        print(f"    │   └── fold10/")
        print(f"    └── metadata/")
        print(f"        └── UrbanSound8K.csv")
        sys.exit(1)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Create configuration
    config = Config()
    
    # Update config with command line arguments
    config.MODEL_NAME = args.model_name
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.NUM_EPOCHS = args.num_epochs
    config.WEIGHT_DECAY = args.weight_decay
    config.SAMPLE_RATE = args.sample_rate
    config.MAX_DURATION = args.max_duration
    config.DROPOUT_RATE = args.dropout_rate
    config.OUTPUT_DIR = args.output_dir
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.LOG_DIR = args.log_dir
    config.DEVICE = str(device)
    config.TRAIN_FOLDS = args.train_folds
    config.VAL_FOLD = args.val_fold
    config.TEST_FOLD = args.test_fold
    
    print("=" * 80)
    print("Wav2Vec2 UrbanSound8K Fine-tuning")
    print("=" * 80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Sample rate: {config.SAMPLE_RATE}")
    print(f"Max duration: {config.MAX_DURATION}s")
    print(f"Train folds: {config.TRAIN_FOLDS}")
    print(f"Val fold: {config.VAL_FOLD}")
    print(f"Test fold: {config.TEST_FOLD}")
    print("=" * 80)
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=config.BATCH_SIZE,
            num_workers=args.num_workers,
            target_sample_rate=config.SAMPLE_RATE,
            max_duration=config.MAX_DURATION,
            train_folds=config.TRAIN_FOLDS,
            val_fold=config.VAL_FOLD,
            test_fold=config.TEST_FOLD
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        freeze_feature_extractor=args.freeze_feature_extractor
    )
    
    # Print model summary
    print_model_summary(model, (config.BATCH_SIZE, int(config.SAMPLE_RATE * config.MAX_DURATION)))
    
    # Create trainer
    trainer = Wav2Vec2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Test only mode
    if args.test_only:
        if not args.resume:
            print("Error: --test_only requires --resume checkpoint")
            sys.exit(1)
        
        print("Running test only...")
        test_results = trainer.test()
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        return
    
    # Start training
    print("Starting training...")
    try:
        test_results = trainer.train()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Test accuracy: {test_results['accuracy']:.4f}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_checkpoint(trainer.best_epoch, is_best=False)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
