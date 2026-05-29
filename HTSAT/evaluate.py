"""
HTSAT Evaluation Script for ESC-50
Based on: https://github.com/RetroCirce/HTS-Audio-Transformer

This script evaluates a pre-trained HTSAT model on ESC-50 dataset
with specified train/validation fold splits.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from htsat_config import config
from htsat_model import load_htsat_checkpoint
from esc50_dataset import get_dataloader


def evaluate_model(model, dataloader, device, class_labels=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: HTSAT model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_labels: List of class label names
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_filenames = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (waveforms, labels, filenames) in enumerate(tqdm(dataloader, desc="Evaluation")):
            # Move to device
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            try:
                logits = model(waveforms)
            except Exception as e:
                print(f"\nError in forward pass for batch {batch_idx}: {e}")
                print(f"Waveform shape: {waveforms.shape}")
                continue
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # Detailed classification report
    if class_labels is not None:
        print("\nPer-Class Performance:")
        print("-" * 60)
        report = classification_report(
            all_labels, all_preds, 
            target_names=class_labels,
            digits=3
        )
        print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Store results
    results = {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'filenames': all_filenames,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            all_labels, all_preds, 
            target_names=class_labels if class_labels else None,
            output_dict=True
        )
    }
    
    return results


def plot_confusion_matrix(cm, class_labels, save_path=None):
    """Plot and optionally save confusion matrix"""
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - HTSAT on ESC-50', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def save_predictions(results, output_dir):
    """Save prediction results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': results['filenames'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'correct': results['labels'] == results['predictions']
    })
    
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.npy')
    np.save(cm_path, results['confusion_matrix'])
    print(f"Confusion matrix saved to {cm_path}")
    
    # Plot and save confusion matrix
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['confusion_matrix'], 
        config.class_labels,
        save_path=cm_plot_path
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate HTSAT on ESC-50')
    parser.add_argument('--checkpoint', type=str, 
                       default=config.pretrained_checkpoint,
                       help='Path to checkpoint file')
    parser.add_argument('--audio_dir', type=str, 
                       default=config.audio_dir,
                       help='Path to audio directory')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to esc50.csv metadata file')
    parser.add_argument('--val_fold', type=int, default=config.val_fold,
                       help='Validation fold number')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=config.num_workers,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found at {args.audio_dir}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("HTSAT EVALUATION ON ESC-50")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Validation fold: {args.val_fold}")
    print(f"Training folds: {config.train_folds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Load model
    print("Loading HTSAT model...")
    model = load_htsat_checkpoint(
        args.checkpoint,
        num_classes=config.num_classes,
        device=device
    )
    
    # Create validation dataloader
    print(f"\nLoading validation data (fold {args.val_fold})...")
    val_loader, val_dataset = get_dataloader(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        target_folds=[args.val_fold],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config.sample_rate,
        clip_samples=config.clip_samples,
        shuffle=False
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        class_labels=config.class_labels
    )
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    save_predictions(results, args.output_dir)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'evaluation_summary_rise.txt')
    with open(summary_path, 'w') as f:
        f.write("HTSAT EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Audio directory: {args.audio_dir}\n")
        f.write(f"Validation fold: {args.val_fold}\n")
        f.write(f"Training folds: {config.train_folds}\n")
        f.write(f"Total validation samples: {len(val_dataset)}\n")
        f.write(f"\nAccuracy: {results['accuracy'] * 100:.2f}%\n")
        f.write("=" * 60 + "\n")
    
    print(f"Summary saved to {summary_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()




