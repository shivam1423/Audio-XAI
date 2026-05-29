"""
HTSAT Evaluation for UrbanSound8K using AudioSet checkpoint
Based on evaluate_htsat.py from HTSAT folder
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import UrbanSound8K dataset loader
from urbansound8k_dataset import get_dataloader
from urbansound8k_config import config as us8k_config

# Import model loader
from load_audioset_model import load_audioset_checkpoint_for_urbansound8k


def evaluate(model, dataloader, device, class_labels):
    """
    Evaluate HTSAT model on UrbanSound8K dataset
    
    Args:
        model: HTSAT model
        dataloader: DataLoader
        device: Device
        class_labels: List of class names
    
    Returns:
        results: Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_filenames = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for waveforms, labels, filenames in tqdm(dataloader, desc="Evaluation"):
            # Move to device
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            try:
                # Forward pass - HTSAT returns dict with 'clipwise_output' and 'framewise_output'
                output_dict = model(waveforms, None, True)  # (x, mixup_lambda, infer_mode)
                
                # Get clipwise predictions
                if isinstance(output_dict, dict):
                    logits = output_dict['clipwise_output']
                else:
                    logits = output_dict
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(filenames)
                
            except Exception as e:
                print(f"\nError in batch: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        print("ERROR: No predictions generated!")
        return None
    
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
            digits=3,
            zero_division=0
        )
        print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'filenames': all_filenames,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate HTSAT AudioSet checkpoint on UrbanSound8K'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to AudioSet checkpoint (.ckpt file)')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Path to UrbanSound8K audio directory (parent of fold1, fold2, etc.)')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to UrbanSound8K.csv metadata file (auto-searched if not provided)')
    parser.add_argument('--test_fold', type=int, default=10,
                       help='Test fold number (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (default: 10 for UrbanSound8K)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("HTSAT AudioSet → UrbanSound8K Evaluation")
    print("="*70)
    print(f"AudioSet Checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Test fold: {args.test_fold}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load model (AudioSet checkpoint adapted for UrbanSound8K)
    model, config = load_audioset_checkpoint_for_urbansound8k(
        args.checkpoint, 
        device=device,
        num_classes=args.num_classes
    )
    
    # Class labels
    class_labels = us8k_config.class_labels
    
    # Load data
    print(f"\nLoading test data (fold {args.test_fold})...")
    test_loader, test_dataset = get_dataloader(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        target_folds=[args.test_fold],
        batch_size=args.batch_size,
        num_workers=us8k_config.num_workers,
        sample_rate=us8k_config.sample_rate,
        clip_samples=us8k_config.clip_samples,
        shuffle=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results = evaluate(model, test_loader, device, class_labels)
    
    if results is None:
        print("Evaluation failed!")
        return
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'filename': results['filenames'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'true_class': [class_labels[l] for l in results['labels']],
        'predicted_class': [class_labels[p] for p in results['predictions']],
        'correct': results['labels'] == results['predictions']
    })
    pred_path = os.path.join(args.output_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved to: {pred_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
    np.save(cm_path, results['confusion_matrix'])
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Save confusion matrix as CSV for easier viewing
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=[f"{i}_{class_labels[i]}" for i in range(len(class_labels))],
        columns=[f"{i}_{class_labels[i]}" for i in range(len(class_labels))]
    )
    cm_csv_path = os.path.join(args.output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"✓ Confusion matrix (CSV) saved to: {cm_csv_path}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("HTSAT AudioSet → UrbanSound8K Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"AudioSet Checkpoint: {args.checkpoint}\n")
        f.write(f"Audio directory: {args.audio_dir}\n")
        f.write(f"Test fold: {args.test_fold}\n")
        f.write(f"Total samples: {len(test_dataset)}\n")
        f.write(f"\nAccuracy: {results['accuracy'] * 100:.2f}%\n")
        f.write("\nPer-Class Accuracy:\n")
        for i, label in enumerate(class_labels):
            class_mask = results['labels'] == i
            if class_mask.sum() > 0:
                class_acc = (results['predictions'][class_mask] == i).mean() * 100
                f.write(f"  {label}: {class_acc:.2f}%\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

