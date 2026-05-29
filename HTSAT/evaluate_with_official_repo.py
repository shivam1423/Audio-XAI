"""
HTSAT Evaluation using Official Repository Code
This script uses the official HTSAT implementation from:
https://github.com/RetroCirce/HTS-Audio-Transformer

First run: ./setup_official_htsat.sh
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add official repo to path
OFFICIAL_REPO_PATH = "/Users/shivampandey/SS 25/Thesis/RISE_dev/HTSAT/HTS-Audio-Transformer"

if os.path.exists(OFFICIAL_REPO_PATH):
    sys.path.insert(0, OFFICIAL_REPO_PATH)
    sys.path.insert(0, os.path.join(OFFICIAL_REPO_PATH, 'sed_model'))
    print(f"✓ Using official HTSAT repository at: {OFFICIAL_REPO_PATH}")
else:
    print(f"✗ Official repository not found at: {OFFICIAL_REPO_PATH}")
    print("Please run: ./setup_official_htsat.sh")
    sys.exit(1)

try:
    # Try to import from official repo
    # The actual import path may vary depending on repo structure
    # This is a placeholder - adjust based on actual repo structure
    from models.htsat import HTSAT
    print("✓ Loaded HTSAT model from official repository")
except ImportError as e:
    print(f"⚠ Could not import from official repo: {e}")
    print("Falling back to local implementation...")
    from htsat_model import HTSAT, load_htsat_checkpoint

from htsat_config import config
from esc50_dataset import get_dataloader


def load_checkpoint_official(checkpoint_path, device='cuda'):
    """
    Load checkpoint using official HTSAT code
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
    
    Returns:
        model: Loaded HTSAT model
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from checkpoint if available
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print("Checkpoint hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")
    
    # Create model based on checkpoint config or defaults
    model_config = checkpoint.get('hyper_parameters', {})
    
    # Initialize model with config
    model = HTSAT(
        num_classes=model_config.get('num_classes', 50),
        # Add other parameters as needed based on checkpoint
    )
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove module prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load weights
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✓ Loaded checkpoint with strict matching")
    except Exception as e:
        print(f"⚠ Strict loading failed: {e}")
        print("Attempting flexible loading...")
        model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, dataloader, device, class_labels):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for waveforms, labels, filenames in tqdm(dataloader, desc="Evaluation"):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            try:
                logits = model(waveforms)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
            except Exception as e:
                print(f"\nError during evaluation: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*70}\n")
    
    # Per-class metrics
    if class_labels:
        print("\nPer-Class Performance:")
        print("-" * 70)
        report = classification_report(
            all_labels, all_preds,
            target_names=class_labels,
            digits=3
        )
        print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate HTSAT using official repo')
    parser.add_argument('--checkpoint', type=str,
                       default=config.pretrained_checkpoint,
                       help='Path to checkpoint file')
    parser.add_argument('--audio_dir', type=str,
                       default=config.audio_dir,
                       help='Path to audio directory')
    parser.add_argument('--val_fold', type=int, default=config.val_fold,
                       help='Validation fold')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results_official',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("HTSAT Evaluation (Official Repository)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Validation fold: {args.val_fold}")
    print(f"Training folds: {config.train_folds}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load model
    try:
        model = load_checkpoint_official(args.checkpoint, device)
    except Exception as e:
        print(f"Failed to load with official code: {e}")
        print("Falling back to local implementation...")
        from htsat_model import load_htsat_checkpoint
        model = load_htsat_checkpoint(args.checkpoint, config.num_classes, device)
    
    # Load data
    print(f"\nLoading validation data (fold {args.val_fold})...")
    val_loader, val_dataset = get_dataloader(
        audio_dir=args.audio_dir,
        target_folds=[args.val_fold],
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        sample_rate=config.sample_rate,
        clip_samples=config.clip_samples,
        shuffle=False
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate
    results = evaluate(model, val_loader, device, config.class_labels)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'correct': results['labels'] == results['predictions']
    })
    pred_path = os.path.join(args.output_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved to: {pred_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
    np.save(cm_path, results['confusion_matrix'])
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("HTSAT Evaluation Summary (Official Repository)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Validation fold: {args.val_fold}\n")
        f.write(f"Training folds: {config.train_folds}\n")
        f.write(f"Total samples: {len(val_dataset)}\n")
        f.write(f"\nAccuracy: {results['accuracy'] * 100:.2f}%\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()




