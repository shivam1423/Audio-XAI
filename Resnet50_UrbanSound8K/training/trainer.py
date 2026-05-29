"""
Training module for ResNet50 UrbanSound8K
Handles training loop, validation, checkpointing, and logging
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np


class Trainer:
    """
    Trainer class for ResNet50 audio classification
    Handles training loop, validation, checkpointing, and CSV logging
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize trainer
        
        Args:
            model: ResNet50 model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
        
        # Learning rate scheduler
        if config.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.scheduler_min_lr,
                verbose=True
            )
        elif config.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.n_epochs
            )
        else:
            self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Checkpoint management
        self.checkpoints = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup CSV logging for training history"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.config.output_dir,
            f'training_log_{timestamp}.csv'
        )
        
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 
                'val_loss', 'val_acc', 'lr', 'time'
            ])
        
        print(f"Logging to: {self.log_file}")
    
    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log epoch results to CSV file"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.8f}", f"{epoch_time:.2f}"
            ])
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_interval == 0:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            
            # Keep only last N checkpoints
            if len(self.checkpoints) > self.config.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
    
    def train(self):
        """
        Main training loop
        
        Returns:
            train_history: Dictionary with training history
            val_history: Dictionary with validation history
        """
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        print(f"Total epochs: {self.config.n_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Log to CSV
            self._log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
            
            # Print epoch summary
            print(f"Epoch [{epoch+1}/{self.config.n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=is_best)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"Best model saved to: {os.path.join(self.config.output_dir, 'best_model.pth')}")
        print(f"Training log saved to: {self.log_file}")
        print("="*70 + "\n")
        
        # Return training history
        train_history = {
            'loss': self.train_losses,
            'accuracy': self.train_accuracies
        }
        
        val_history = {
            'loss': self.val_losses,
            'accuracy': self.val_accuracies
        }
        
        return train_history, val_history
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
