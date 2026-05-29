"""
Evaluator for ACDNet on UrbanSound8K
Implements 10-crop testing strategy as per paper
"""

import os
import sys
import math
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.acdnet import GetACDNetModel


class ACDNetEvaluator:
    """
    Evaluator for ACDNet on UrbanSound8K
    
    Implements 10-crop testing using pre-generated multi-crop data:
    - Load pre-processed 10-crop test data
    - Forward pass all crops through model
    - Average predictions across all crops per sample
    - Report accuracy, confusion matrix, and per-class metrics
    
    Follows original ACDNet evaluation methodology
    """
    
    def __init__(self, config, model_path):
        """
        Initialize evaluator
        
        Args:
            config: ACDNetConfig object
            model_path: Path to trained model checkpoint
        """
        self.config = config
        self.model_path = model_path
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = self.load_model()
        self.model.eval()
        
        # Load pre-generated multi-crop test data
        test_data_path = f'./val_data/fold{config.test_fold}_val10crop.npz'
        print(f"\nLoading multi-crop test data from: {test_data_path}")
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"Multi-crop test data not found: {test_data_path}\n"
                f"Please run: python scripts/prepare_validation_data.py "
                f"--npz_path {config.npz_path} --output_dir ./val_data"
            )
        
        data = np.load(test_data_path, allow_pickle=True)
        
        # Apply moveaxis to convert from (n, 1, length, 1) to (n, 1, 1, length)
        self.test_x = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.device)
        self.test_y = torch.tensor(data['y']).to(self.device)
        
        n_samples = len(self.test_x) // config.n_crops
        print(f"  Loaded {len(self.test_x)} crops ({n_samples} samples × {config.n_crops} crops)")
        print(f"  test_x shape: {self.test_x.shape}")
        print(f"  test_y shape: {self.test_y.shape}")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        model = GetACDNetModel(
            input_len=self.config.input_length,
            nclass=self.config.num_classes,
            sr=self.config.sr
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_val_acc' in checkpoint:
            print(f"  Best val accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        return model
    
    def evaluate(self):
        """
        Evaluate model on test set with multi-crop averaging
        Following original ACDNet pattern
        
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*70)
        print("Evaluating ACDNet on UrbanSound8K Test Set")
        print("="*70)
        print(f"Test fold: {self.config.test_fold}")
        print(f"Number of crops per sample: {self.config.n_crops}")
        print("="*70 + "\n")
        
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass on all crops (batched for efficiency)
            y_pred = None
            batch_size = (self.config.batch_size // self.config.n_crops) * self.config.n_crops
            
            print("Running inference on all test crops...")
            for idx in range(math.ceil(len(self.test_x) / batch_size)):
                x = self.test_x[idx*batch_size : (idx+1)*batch_size]
                scores = self.model(x)
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data))
                
                if (idx + 1) % 10 == 0 or idx == math.ceil(len(self.test_x) / batch_size) - 1:
                    processed = min((idx + 1) * batch_size, len(self.test_x))
                    n_samples = len(self.test_x) // self.config.n_crops
                    print(f"  Processed {processed}/{len(self.test_x)} crops ({processed//self.config.n_crops}/{n_samples} samples)")
            
            print("\nAveraging predictions across crops...")
            # Reshape and average predictions across 10 crops per sample
            n_samples = y_pred.shape[0] // self.config.n_crops
            y_pred_reshaped = y_pred.reshape(n_samples, self.config.n_crops, y_pred.shape[1])
            y_target_reshaped = self.test_y.reshape(n_samples, self.config.n_crops, self.test_y.shape[1])
            
            # Average across crops
            y_pred_avg = y_pred_reshaped.mean(dim=1).argmax(dim=1)
            y_target_avg = y_target_reshaped.mean(dim=1).argmax(dim=1)
            
            # Convert to numpy for sklearn metrics
            all_preds = y_pred_avg.cpu().numpy()
            all_labels = y_target_avg.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Per-class metrics
        class_report = classification_report(
            all_labels,
            all_preds,
            target_names=self.config.class_labels,
            digits=4,
            output_dict=True
        )
        
        # Prepare results
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': all_preds,
            'labels': all_labels,
            'test_fold': self.config.test_fold,
            'n_samples': n_samples,
            'n_crops': self.config.n_crops
        }
        
        # Print results
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Number of test samples: {n_samples}")
        print("\nPer-Class Performance:")
        print("-"*70)
        
        for class_name in self.config.class_labels:
            metrics = class_report[class_name]
            print(
                f"{class_name:20s} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1-score']:.4f} | "
                f"Support: {metrics['support']:4.0f}"
            )
        
        print("-"*70)
        print(f"Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}")
        print("="*70 + "\n")
        
        return results
    
    def save_results(self, results, output_dir):
        """
        Save evaluation results to disk
        
        Args:
            results: Dictionary with evaluation results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'classification_report': results['classification_report'],
            'test_fold': results['test_fold'],
            'n_samples': results['n_samples'],
            'n_crops': results['n_crops']
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Results saved to: {results_path}")
        
        # Save predictions and labels as numpy arrays
        preds_path = os.path.join(output_dir, 'predictions.npy')
        labels_path = os.path.join(output_dir, 'labels.npy')
        
        np.save(preds_path, np.array(results['predictions']))
        np.save(labels_path, np.array(results['labels']))
        
        print(f"✓ Predictions saved to: {preds_path}")
        print(f"✓ Labels saved to: {labels_path}")
        
        # Save confusion matrix
        conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.npy')
        np.save(conf_matrix_path, np.array(results['confusion_matrix']))
        print(f"✓ Confusion matrix saved to: {conf_matrix_path}")


def evaluate_model(config, model_path, output_dir):
    """
    Convenience function to evaluate a trained model
    
    Args:
        config: ACDNetConfig object
        model_path: Path to model checkpoint
        output_dir: Directory to save results
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = ACDNetEvaluator(config, model_path)
    results = evaluator.evaluate()
    evaluator.save_results(results, output_dir)
    return results
