"""
Fine-tune HTSAT AudioSet checkpoint on UrbanSound8K
Transfer learning: Use AudioSet features, train new 10-class head
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import time

# Import UrbanSound8K dataset loader
from urbansound8k_dataset import get_dataloader
from urbansound8k_config import config as us8k_config

# Import model loader
from load_audioset_model import load_audioset_checkpoint_for_urbansound8k


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (waveforms, labels, _) in enumerate(pbar):
        # Move to device
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # HTSAT returns dict with 'clipwise_output'
            output_dict = model(waveforms, None, False)  # training mode
            
            if isinstance(output_dict, dict):
                logits = output_dict['clipwise_output']
            else:
                logits = output_dict
            
            # Loss
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            try:
                # Forward pass
                output_dict = model(waveforms, None, True)  # inference mode
                
                if isinstance(output_dict, dict):
                    logits = output_dict['clipwise_output']
                else:
                    logits = output_dict
                
                # Loss
                loss = criterion(logits, labels)
                
                # Metrics
                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"\nError in validation: {e}")
                continue
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune HTSAT AudioSet checkpoint on UrbanSound8K'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to AudioSet checkpoint (.ckpt file)')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Path to UrbanSound8K audio directory')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to UrbanSound8K.csv metadata file')
    parser.add_argument('--test_fold', type=int, default=10,
                       help='Test fold number (default: 10)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16 for training)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--freeze_features', action='store_true',
                       help='Freeze all layers except classification head')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (default: 10)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("HTSAT Fine-tuning on UrbanSound8K")
    print("="*70)
    print(f"AudioSet Checkpoint: {args.checkpoint}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Test fold: {args.test_fold}")
    print(f"Training folds: {[f for f in range(1, 11) if f != args.test_fold]}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Freeze features: {args.freeze_features}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading AudioSet checkpoint and adapting for UrbanSound8K...")
    model, config = load_audioset_checkpoint_for_urbansound8k(
        args.checkpoint,
        device=device,
        num_classes=args.num_classes
    )
    
    # Freeze features if requested
    if args.freeze_features:
        print("\nFreezing feature extraction layers...")
        for name, param in model.named_parameters():
            # Freeze everything except head and tscam_conv
            if 'head' not in name and 'tscam_conv' not in name:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    else:
        print("\nTraining all layers (full fine-tuning)")
    
    # Prepare data
    print("\nPreparing data...")
    train_folds = [f for f in range(1, 11) if f != args.test_fold]
    
    train_loader, train_dataset = get_dataloader(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        target_folds=train_folds,
        batch_size=args.batch_size,
        num_workers=us8k_config.num_workers,
        sample_rate=us8k_config.sample_rate,
        clip_samples=us8k_config.clip_samples,
        shuffle=True  # Important for training
    )
    
    val_loader, val_dataset = get_dataloader(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        target_folds=[args.test_fold],
        batch_size=args.batch_size,
        num_workers=us8k_config.num_workers,
        sample_rate=us8k_config.sample_rate,
        clip_samples=us8k_config.clip_samples,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc*100:.2f}%)")
        
        # Save latest checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"Total training time: {elapsed_time/60:.1f} minutes")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to: {history_path}")
    
    # Final evaluation on validation set with best model
    print("\n" + "="*70)
    print("Final Evaluation (Best Model)")
    print("="*70)
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device, "Final"
    )
    
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    print("\nPer-Class Performance:")
    print("-" * 70)
    report = classification_report(
        val_labels, val_preds,
        target_names=us8k_config.class_labels,
        digits=3
    )
    print(report)
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write("HTSAT Fine-tuning Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"AudioSet Checkpoint: {args.checkpoint}\n")
        f.write(f"Test fold: {args.test_fold}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation accuracy: {best_val_acc*100:.2f}%\n\n")
        f.write("Per-Class Performance:\n")
        f.write("-" * 70 + "\n")
        f.write(report)
    
    print(f"\n✓ Final results saved to: {results_path}")
    print(f"✓ Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print("\nTraining pipeline complete!")


if __name__ == '__main__':
    main()

