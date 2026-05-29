"""
Utility functions for Wav2Vec2 UrbanSound8K project
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def plot_audio_samples(
    audio_samples: List[torch.Tensor], 
    labels: List[int], 
    class_names: List[str],
    sample_rate: int = 16000,
    save_path: Optional[str] = None
):
    """
    Plot audio samples with their labels
    
    Args:
        audio_samples: List of audio tensors
        labels: List of corresponding labels
        class_names: List of class names
        sample_rate: Sample rate of audio
        save_path: Optional path to save the plot
    """
    n_samples = len(audio_samples)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, (audio, label) in enumerate(zip(audio_samples, labels)):
        time_axis = torch.linspace(0, len(audio) / sample_rate, len(audio))
        
        axes[i].plot(time_axis, audio.numpy())
        axes[i].set_title(f"Class: {class_names[label]}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(
    labels: List[int], 
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None
):
    """
    Plot class distribution
    
    Args:
        labels: List of labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique_labels)), counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(range(len(unique_labels)), [class_names[i] for i in unique_labels], rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_predictions(
    predictions: List[int],
    labels: List[int],
    probabilities: List[List[float]],
    class_names: List[str],
    save_path: str
):
    """
    Save predictions to CSV file
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        probabilities: List of prediction probabilities
        class_names: List of class names
        save_path: Path to save the CSV file
    """
    import pandas as pd
    
    data = []
    for i, (pred, true, probs) in enumerate(zip(predictions, labels, probabilities)):
        row = {
            'sample_id': i,
            'true_label': true,
            'predicted_label': pred,
            'true_class': class_names[true],
            'predicted_class': class_names[pred],
            'correct': pred == true
        }
        
        # Add probabilities for each class
        for j, prob in enumerate(probs):
            row[f'prob_{class_names[j]}'] = prob
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict, config_path: str):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_directory_structure(base_dir: str):
    """
    Create directory structure for the project
    
    Args:
        base_dir: Base directory for the project
    """
    directories = [
        "outputs",
        "checkpoints",
        "logs",
        "results"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")
    
    print("\nModel Architecture:")
    print("-" * 40)
    
    # Print model structure
    def print_module(module, prefix=""):
        for name, child in module.named_children():
            if len(list(child.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in child.parameters())
                print(f"{prefix}{name}: {child.__class__.__name__} ({params:,} params)")
            else:
                print(f"{prefix}{name}: {child.__class__.__name__}")
                print_module(child, prefix + "  ")
    
    print_module(model)
    
    print("=" * 80)
