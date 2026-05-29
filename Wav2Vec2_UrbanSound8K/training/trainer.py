"""
Training module for Wav2Vec2 UrbanSound8K fine-tuning
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from model.wav2vec2_classifier import Wav2Vec2Classifier
from config import Config


class Wav2Vec2Trainer:
    """
    Trainer class for Wav2Vec2 fine-tuning on UrbanSound8K
    """
    
    def __init__(
        self,
        model: Wav2Vec2Classifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
        device: str = "cuda"
    ):
        """
        Initialize trainer
        
        Args:
            model: Wav2Vec2 classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration object
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Create output directories
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different parts"""
        # Different learning rates for Wav2Vec2 and classifier
        wav2vec2_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'wav2vec2' in name:
                wav2vec2_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': wav2vec2_params, 'lr': self.config.LEARNING_RATE * 0.1},  # Lower LR for pre-trained
            {'params': classifier_params, 'lr': self.config.LEARNING_RATE}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.NUM_EPOCHS,
            eta_min=self.config.LEARNING_RATE * 0.01
        )
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to TensorBoard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for audio, labels in tqdm(self.val_loader, desc="Validating"):
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(audio)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def test(self) -> Dict:
        """Test the model and return detailed metrics"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for audio, labels in tqdm(self.test_loader, desc="Testing"):
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(audio)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=self.config.CLASS_NAMES,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model_wav2vec2.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            # Restore config values if dict stored
            if isinstance(checkpoint.get('config'), dict):
                for k, v in checkpoint['config'].items():
                    setattr(self.config, k, v)
            
            return checkpoint['epoch']
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return -1  # Indicate failure
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'training_history.png'))
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix"""
        # Adjusted size for 10 classes (UrbanSound8K)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Best Val Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})")
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        # Plot training history
        self.plot_training_history()
        
        # Load best model for testing
        best_checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model_wav2vec2.pt')
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
            print("Loaded best model for testing")
        
        # Test
        print("\nTesting on test set...")
        test_results = self.test()
        
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(test_results['confusion_matrix'], self.config.CLASS_NAMES)
        
        # Save test results
        import json
        with open(os.path.join(self.config.OUTPUT_DIR, 'test_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {
                'accuracy': test_results['accuracy'],
                'classification_report': test_results['classification_report']
            }
            json.dump(results_to_save, f, indent=2)
        
        self.writer.close()
        
        return test_results
