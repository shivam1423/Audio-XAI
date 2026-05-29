"""
HTSAT Evaluation using Official Architecture
Fix for device mismatch and model loading errors
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
OFFICIAL_REPO = "HTS-Audio-Transformer"
sys.path.insert(0, OFFICIAL_REPO)

# Import official HTSAT model
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper

# Import dataset loader
sys.path.insert(0, "HTSAT")
from esc50_dataset import get_dataloader


class HTSATConfig:
    """Configuration matching the official HTSAT ESC-50 setup"""

    # Dataset config
    dataset_type = "esc-50"
    classes_num = 50

    # Loss config
    loss_type = "clip_ce"  # ESC-50 uses clip_ce

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
    clip_samples = 32000 * 10  # 10 seconds
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000

    # Model features
    enable_tscam = True

    # Deprecated optimization flags (from official esc_config.py)
    htsat_attn_heatmap = False
    htsat_hier_output = False
    htsat_use_max = False

    # Data augmentation flags
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False

    # ESC-50 specific
    val_fold = 1
    train_folds = [2, 3, 4, 5]

    # Evaluation
    batch_size = 32
    num_workers = 4


def load_htsat_model(checkpoint_path, device='cuda'):
    """
    Load HTSAT model from checkpoint using official architecture
    
    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to load on
    
    Returns:
        model: Loaded HTSAT model (sed_model component)
    """
    config = HTSATConfig()
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create HTSAT model with exact same config as training
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=config.classes_num,
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_head=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Extract sed_model weights
    sed_model_state = {}
    for key, value in state_dict.items():
        if key.startswith('sed_model.'):
            # Remove 'sed_model.' prefix
            new_key = key[10:]
            sed_model_state[new_key] = value
    
    # Load weights
    missing_keys, unexpected_keys = sed_model.load_state_dict(sed_model_state, strict=False)
    
    if len(missing_keys) > 0:
        print(f"Warning: {len(missing_keys)} missing keys")
        if len(missing_keys) < 20:
            for key in missing_keys:
                print(f"  - {key}")
    
    if len(unexpected_keys) > 0:
        print(f"Warning: {len(unexpected_keys)} unexpected keys") 
        if len(unexpected_keys) < 20:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    # Move model to device BEFORE eval mode to ensure all buffers are moved
    sed_model = sed_model.to(device)
    
    # Ensure all spectrogram extractor components are on the correct device
    if hasattr(sed_model, 'spectrogram_extractor'):
        sed_model.spectrogram_extractor = sed_model.spectrogram_extractor.to(device)
    if hasattr(sed_model, 'logmel_extractor'):
        sed_model.logmel_extractor = sed_model.logmel_extractor.to(device)
    
    sed_model.eval()
    
    print(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    return sed_model, config


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
    parser = argparse.ArgumentParser(description='Evaluate HTSAT on ESC-50')
    parser.add_argument('--checkpoint', type=str,
                       default='HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt',
                       help='Path to checkpoint')
    parser.add_argument('--audio_dir', type=str,
                       default='../ESC50/audio',
                       help='Path to audio directory')
    parser.add_argument('--val_fold', type=int, default=2,
                       help='Validation fold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results_fold2',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("HTSAT Evaluation (Official Architecture)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Validation fold: {args.val_fold}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load model
    model, config = load_htsat_model(args.checkpoint, device)
    
    # Class labels
    class_labels = [
        'dog', 'rooster', 'pig', 'cow', 'frog',
        'cat', 'hen', 'insects', 'sheep', 'crow',
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
        'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
        'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
        'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
        'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
        'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
        'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
        'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
    ]
    
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
    results = evaluate(model, val_loader, device, class_labels)
    
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
        f.write("HTSAT Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Validation fold: {args.val_fold}\n")
        f.write(f"Total samples: {len(val_dataset)}\n")
        f.write(f"\nAccuracy: {results['accuracy'] * 100:.2f}%\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()



