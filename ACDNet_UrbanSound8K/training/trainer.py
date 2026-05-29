"""
Trainer for ACDNet UrbanSound8K Classification
Implements training loop with BC Learning, LR scheduling, and validation
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.acdnet import GetACDNetModel, get_model_summary
from training.train_generator import setup_generator
from utils.helpers import to_hms


class ACDNetTrainer:
    """
    Trainer class for ACDNet on UrbanSound8K
    
    Implements:
    - BC (Between-Class) Learning with NPZ data loading
    - Learning rate scheduling
    - Validation during training
    - Model checkpointing
    
    Follows original ACDNet implementation pattern
    """
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: ACDNetConfig object with all training parameters
        """
        self.config = config
        
        # Validate configuration
        config.validate()
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load preprocessed NPZ file
        print(f"\nLoading preprocessed dataset from NPZ: {config.npz_path}")
        if not os.path.exists(config.npz_path):
            raise FileNotFoundError(
                f"NPZ file not found: {config.npz_path}\n"
                f"Please run: python scripts/prepare_urbansound8k.py --data_dir <path> --output_dir ./data"
            )
        
        self.dataset = np.load(config.npz_path, allow_pickle=True)
        
        # Create BC Learning generator for training
        print("\nInitializing BC Learning generator...")
        self.train_generator = setup_generator(
            npz_path=config.npz_path,
            config=config,
            train_folds=config.train_folds
        )
        
        print(f"\nDataset loaded:")
        print(f"  - Training folds: {config.train_folds} ({len(self.train_generator) * config.batch_size} samples)")
        print(f"  - Validation fold: {config.val_fold}")
        print(f"  - Test fold: {config.test_fold}")
        
        # Multi-crop validation data (loaded lazily during first validation)
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        
        # Create model
        print("\nInitializing ACDNet model...")
        self.model = GetACDNetModel(
            input_len=config.input_length,
            nclass=config.num_classes,
            sr=config.sr
        ).to(self.device)
        
        # Print model summary
        summary = get_model_summary(self.model)
        print(f"Model Parameters: {summary['total_parameters']:,}")
        print(f"Model Size: {summary['model_size_mb']:.2f} MB")
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []
        
        # Create output directory
        os.makedirs(config.trained_models_dir, exist_ok=True)
    
    def load_validation_data(self):
        """
        Load pre-generated multi-crop validation data
        Following original ACDNet pattern from torch/trainer.py:113-115
        
        Loads from: val_data/fold{val_fold}_val10crop.npz
        Format: x shape (n_samples*10, 1, input_length, 1)
                y shape (n_samples*10, num_classes)
        """
        val_data_path = f'./val_data/fold{self.config.val_fold}_val10crop.npz'
        
        if not os.path.exists(val_data_path):
            raise FileNotFoundError(
                f"Multi-crop validation data not found: {val_data_path}\n"
                f"Please run: python scripts/prepare_validation_data.py "
                f"--npz_path {self.config.npz_path} --output_dir ./val_data"
            )
        
        print(f"\nLoading multi-crop validation data from: {val_data_path}")
        data = np.load(val_data_path, allow_pickle=True)
        
        # Apply moveaxis to convert from (n, 1, length, 1) to (n, 1, 1, length)
        # This is the same transformation used in training
        self.val_x = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.device)
        self.val_y = torch.tensor(data['y']).to(self.device)
        
        n_samples = len(self.val_x) // self.config.n_crops
        print(f"  Loaded {len(self.val_x)} crops ({n_samples} samples × {self.config.n_crops} crops)")
        print(f"  val_x shape: {self.val_x.shape}")
        print(f"  val_y shape: {self.val_y.shape}")
    
    def get_lr(self, epoch):
        """
        Calculate learning rate for current epoch
        
        Args:
            epoch: Current epoch number (1-indexed)
        
        Returns:
            Learning rate for this epoch
        """
        # Calculate decay schedule
        divide_epoch = np.array([self.config.n_epochs * i for i in self.config.schedule])
        decay = sum(epoch > divide_epoch)
        
        # Warmup phase
        if epoch <= self.config.warmup:
            decay = 1
        
        lr = self.config.lr * np.power(0.1, decay)
        return lr
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            train_loss, train_acc: Average loss and accuracy for this epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        
        # Number of batches
        n_batches = len(self.train_generator)
        
        for batch_idx in range(n_batches):
            # Get batch from BC generator (returns numpy arrays)
            x, y = self.train_generator[batch_idx]
            
            # Apply moveaxis to convert from (batch, 1, length, 1) to (batch, 1, 1, length)
            # This is CRITICAL for ACDNet architecture!
            x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.device)
            y = torch.tensor(y).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Calculate loss (KL Divergence for soft labels)
            loss = self.loss_func(outputs.log(), y)
            
            # Calculate accuracy
            acc = ((outputs.data.argmax(dim=1) == y.argmax(dim=1)) * 1).float().mean().item()
            running_acc += acc
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate epoch statistics
        train_loss = running_loss / n_batches
        train_acc = (running_acc / n_batches) * 100
        
        return train_loss, train_acc
    
    def validate(self):
        """
        Validate on validation set using multi-crop evaluation
        Following original ACDNet pattern from torch/trainer.py:129-144
        
        Process:
        1. Load pre-generated multi-crop validation data (if not loaded)
        2. Forward pass on all crops (batched)
        3. Reshape predictions from (n_samples*10, num_classes) to (n_samples, 10, num_classes)
        4. Average across 10 crops per sample
        5. Calculate accuracy
        
        Returns:
            val_loss, val_acc: Average validation loss and accuracy
        """
        # Load multi-crop validation data if not already loaded
        if self.val_x is None:
            self.load_validation_data()
        
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass on all crops (batched for efficiency)
            y_pred = None
            # Ensure batch size is divisible by n_crops
            batch_size = (self.config.batch_size // self.config.n_crops) * self.config.n_crops
            
            for idx in range(math.ceil(len(self.val_x) / batch_size)):
                x = self.val_x[idx*batch_size : (idx+1)*batch_size]
                scores = self.model(x)
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data))
            
            # Reshape and average predictions across 10 crops per sample
            # From: (n_samples*10, num_classes)
            # To: (n_samples, 10, num_classes) -> mean -> (n_samples, num_classes)
            n_samples = y_pred.shape[0] // self.config.n_crops
            y_pred_reshaped = y_pred.reshape(n_samples, self.config.n_crops, y_pred.shape[1])
            y_target_reshaped = self.val_y.reshape(n_samples, self.config.n_crops, self.val_y.shape[1])
            
            # Average across crops (dim=1)
            y_pred_avg = y_pred_reshaped.mean(dim=1).argmax(dim=1)
            y_target_avg = y_target_reshaped.mean(dim=1).argmax(dim=1)
            
            # Calculate accuracy
            acc = ((y_pred_avg == y_target_avg).float().mean() * 100).item()
            
            # Calculate loss on averaged predictions
            y_pred_avg_probs = y_pred_reshaped.mean(dim=1)
            y_target_avg_probs = y_target_reshaped.mean(dim=1)
            loss = self.loss_func(y_pred_avg_probs.log(), y_target_avg_probs).item()
        
        self.model.train()
        
        return loss, acc
    
    def save_model(self, epoch, val_acc):
        """
        Save model checkpoint if it's the best so far
        
        Args:
            epoch: Current epoch number
            val_acc: Validation accuracy
        """
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
            # Remove old best model
            old_model_path = os.path.join(
                self.config.trained_models_dir,
                'acdnet_us8k_best.pt'
            )
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
            
            # Save new best model
            model_path = os.path.join(
                self.config.trained_models_dir,
                'acdnet_us8k_best.pt'
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'config': self.config.__dict__
            }, model_path)
            
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting ACDNet Training on UrbanSound8K")
        print("="*70)
        self.config.display()
        print("="*70 + "\n")
        
        # Setup loss and optimizer
        self.loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
            nesterov=True
        )
        
        # Training loop
        train_start_time = time.time()
        
        for epoch in range(1, self.config.n_epochs + 1):
            epoch_start_time = time.time()
            
            # Update learning rate
            current_lr = self.get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            epoch_train_time = time.time() - epoch_start_time
            
            # Validate
            val_loss, val_acc = self.validate()
            epoch_time = time.time() - epoch_start_time
            val_time = epoch_time - epoch_train_time
            
            # Save best model
            self.save_model(epoch, val_acc)
            
            # Store history
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'acc': train_acc
            })
            self.val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                'acc': val_acc
            })
            
            # Print epoch summary
            print(
                f"Epoch: {epoch:3d}/{self.config.n_epochs} | "
                f"Time: {to_hms(epoch_time)} (Train {to_hms(epoch_train_time)} | Val {to_hms(val_time)}) | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% | "
                f"Best: {self.best_val_acc:.2f}%@{self.best_epoch}"
            )
        
        # Training complete
        total_time = time.time() - train_start_time
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total training time: {to_hms(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        print(f"Best model saved to: {self.config.trained_models_dir}/acdnet_us8k_best.pt")
        print("="*70 + "\n")
        
        return self.train_history, self.val_history
