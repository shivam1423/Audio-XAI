"""
Evaluation module for ResNet50 UrbanSound8K
Computes metrics, confusion matrix, and per-class performance
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Evaluator class for ResNet50 audio classification
    Computes detailed metrics and generates visualizations
    """
    
    def __init__(self, model, test_loader, config, device=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained ResNet50 model
            test_loader: Test data loader
            config: Configuration object
            device: Device to run evaluation on (defaults to config.device)
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        
        # Device setup
        if device is None:
            device = config.device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Evaluator initialized on device: {self.device}")
    
    def evaluate(self):
        """
        Run evaluation on test set
        
        Returns:
            results: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*70)
        print("Running Evaluation")
        print("="*70)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                # Store results
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        results = self._compute_metrics(all_preds, all_labels, all_probs)
        
        # Print summary
        self._print_summary(results)
        
        print("="*70 + "\n")
        
        return results
    
    def _compute_metrics(self, preds, labels, probs):
        """
        Compute comprehensive evaluation metrics
        
        Args:
            preds: Predicted class labels
            labels: True class labels
            probs: Prediction probabilities
            
        Returns:
            results: Dictionary with all metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Per-class accuracy (diagonal of normalized confusion matrix)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        per_class_accuracy = np.diag(cm_normalized)
        
        # Organize results
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'total_samples': len(labels)
            },
            'per_class': {},
            'confusion_matrix': cm.tolist(),
            'predictions': preds.tolist(),
            'labels': labels.tolist(),
            'probabilities': probs.tolist()
        }
        
        # Add per-class metrics
        for i in range(self.config.num_classes):
            class_name = self.config.class_labels[i]
            results['per_class'][class_name] = {
                'class_id': i,
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'accuracy': float(per_class_accuracy[i]),
                'support': int(support[i])
            }
        
        return results
    
    def _print_summary(self, results):
        """Print evaluation summary"""
        print("\nOverall Metrics:")
        print(f"  Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"  Precision (macro): {results['overall']['precision_macro']:.4f}")
        print(f"  Recall (macro): {results['overall']['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {results['overall']['f1_macro']:.4f}")
        print(f"  Total samples: {results['overall']['total_samples']}")
        
        print("\nPer-Class Performance:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'Support':<10}")
        print("-" * 70)
        for class_name, metrics in results['per_class'].items():
            print(f"{class_name:<20} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['support']:<10}")
    
    def save_results(self, results, output_dir=None):
        """
        Save evaluation results to files
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory (defaults to config.results_dir)
        """
        if output_dir is None:
            output_dir = self.config.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results as JSON
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to: {results_path}")
        
        # Save confusion matrix
        cm = np.array(results['confusion_matrix'])
        cm_path = os.path.join(output_dir, 'confusion_matrix.npy')
        np.save(cm_path, cm)
        print(f"✓ Saved confusion matrix to: {cm_path}")
        
        # Save predictions
        preds_path = os.path.join(output_dir, 'predictions.npy')
        np.save(preds_path, np.array(results['predictions']))
        print(f"✓ Saved predictions to: {preds_path}")
        
        # Save labels
        labels_path = os.path.join(output_dir, 'labels.npy')
        np.save(labels_path, np.array(results['labels']))
        print(f"✓ Saved labels to: {labels_path}")
        
        # Generate and save visualizations
        self.plot_confusion_matrix(results, output_dir)
        self.plot_per_class_metrics(results, output_dir)
    
    def plot_confusion_matrix(self, results, output_dir):
        """
        Plot and save confusion matrix
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
        """
        cm = np.array(results['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.config.class_labels,
                   yticklabels=self.config.class_labels)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Plot normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                   xticklabels=self.config.class_labels,
                   yticklabels=self.config.class_labels)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save figure
        cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix plot to: {cm_plot_path}")
    
    def plot_per_class_metrics(self, results, output_dir):
        """
        Plot per-class metrics
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
        """
        class_names = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, metrics in results['per_class'].items():
            class_names.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
        
        # Create bar plot
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        metrics_plot_path = os.path.join(output_dir, 'per_class_metrics.png')
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved per-class metrics plot to: {metrics_plot_path}")


def load_model_for_evaluation(checkpoint_path, config):
    """
    Load trained model from checkpoint for evaluation
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        
    Returns:
        model: Loaded model in eval mode
    """
    from models.resnet50 import create_model
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_acc' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model
