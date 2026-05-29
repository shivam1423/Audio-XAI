"""
Evaluate a fine-tuned HTSAT model on UrbanSound8K
This script loads trained .pth checkpoints (not AudioSet .ckpt)
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

# Add official repo to path
OFFICIAL_REPO = "../HTSAT/HTS-Audio-Transformer"
if os.path.exists(OFFICIAL_REPO):
    sys.path.insert(0, OFFICIAL_REPO)
else:
    OFFICIAL_REPO = "HTS-Audio-Transformer"
    if os.path.exists(OFFICIAL_REPO):
        sys.path.insert(0, OFFICIAL_REPO)

# Import official HTSAT model
from model.htsat import HTSAT_Swin_Transformer

# Import dataset loader
from urbansound8k_dataset import get_dataloader
from urbansound8k_config import config as us8k_config


class HTSATConfig:
    """Configuration for HTSAT UrbanSound8K model"""

    # Dataset config
    dataset_type = "urbansound8k"
    classes_num = 10

    # Loss config
    loss_type = "clip_ce"

    # Model architecture config
    htsat_window_size = 8
    htsat_spec_size = 256
    htsat_patch_size = 4
    htsat_stride = (4, 4)
    htsat_num_head = [4, 8, 16, 32]
    htsat_dim = 96
    htsat_depth = [2, 2, 6, 2]

    # Audio processing
    sample_rate = 32000
    clip_samples = 32000 * 10
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000

    # Model features
    enable_tscam = True

    # Deprecated optimization flags (required by HTSAT model)
    htsat_attn_heatmap = False
    htsat_hier_output = False
    htsat_use_max = False

    # Data augmentation flags (required by HTSAT model)
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False

    # Evaluation
    batch_size = 32
    num_workers = 4


def load_trained_model(checkpoint_path, num_classes=10, device='cuda'):
    """
    Load a fine-tuned HTSAT model from .pth checkpoint
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_classes: Number of classes (10 for UrbanSound8K)
        device: Device to load on
    
    Returns:
        model: Loaded HTSAT model
        checkpoint_info: Dictionary with checkpoint metadata
    """
    config = HTSATConfig()
    config.classes_num = num_classes
    
    print(f"Loading trained model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create HTSAT model
    print(f"Creating HTSAT model ({num_classes} classes)...")
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=num_classes,
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_head=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )
    
    # Load trained weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Loading model_state_dict from checkpoint...")
    else:
        state_dict = checkpoint
        print("Loading state_dict directly from checkpoint...")
    
    # Load weights
    missing_keys, unexpected_keys = sed_model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"Warning: {len(missing_keys)} missing keys")
        if len(missing_keys) < 20:
            for key in missing_keys[:10]:
                print(f"  - {key}")
    
    if len(unexpected_keys) > 0:
        print(f"Warning: {len(unexpected_keys)} unexpected keys")
        if len(unexpected_keys) < 20:
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
    
    # Move model to device
    sed_model = sed_model.to(device)
    
    # Ensure spectrogram extractors are on correct device
    if hasattr(sed_model, 'spectrogram_extractor'):
        sed_model.spectrogram_extractor = sed_model.spectrogram_extractor.to(device)
    if hasattr(sed_model, 'logmel_extractor'):
        sed_model.logmel_extractor = sed_model.logmel_extractor.to(device)
    
    sed_model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_acc': checkpoint.get('val_acc', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown')
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
    if checkpoint_info['val_acc'] != 'unknown':
        print(f"  Checkpoint val accuracy: {checkpoint_info['val_acc']*100:.2f}%")
    
    return sed_model, config, checkpoint_info


def evaluate(model, dataloader, device, class_labels):
    """
    Evaluate HTSAT model on dataset
    
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
        description='Evaluate fine-tuned HTSAT model on UrbanSound8K'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Path to UrbanSound8K audio directory')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to UrbanSound8K.csv metadata file')
    parser.add_argument('--test_fold', type=int, default=10,
                       help='Test fold number (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results_trained',
                       help='Output directory')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (default: 10)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("HTSAT Trained Model Evaluation on UrbanSound8K")
    print("="*70)
    print(f"Trained checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Test fold: {args.test_fold}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load trained model
    model, config, checkpoint_info = load_trained_model(
        args.checkpoint,
        num_classes=args.num_classes,
        device=device
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
    
    # Save confusion matrix as CSV
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
        f.write("HTSAT Trained Model Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Trained checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint epoch: {checkpoint_info['epoch']}\n")
        if checkpoint_info['val_acc'] != 'unknown':
            f.write(f"Training val accuracy: {checkpoint_info['val_acc']*100:.2f}%\n")
        f.write(f"Audio directory: {args.audio_dir}\n")
        f.write(f"Test fold: {args.test_fold}\n")
        f.write(f"Total samples: {len(test_dataset)}\n")
        f.write(f"\nTest Accuracy: {results['accuracy'] * 100:.2f}%\n")
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

