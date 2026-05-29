"""
Utility helper functions for ResNet50 UrbanSound8K training
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to: {seed}")


def get_device(device_name='cuda'):
    """
    Get computing device (CPU or GPU)
    
    Args:
        device_name: Preferred device ('cuda' or 'cpu')
        
    Returns:
        device: PyTorch device
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device


def count_parameters(model):
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds):
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        formatted_time: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_config(config, output_dir):
    """
    Save configuration to file
    
    Args:
        config: Configuration object
        output_dir: Output directory
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert config to dictionary
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    
    # Save as JSON
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Saved configuration to: {config_path}")


def create_output_dirs(config):
    """
    Create necessary output directories
    
    Args:
        config: Configuration object
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    print(f"✓ Output directories created:")
    print(f"  - Models: {config.output_dir}")
    print(f"  - Results: {config.results_dir}")


def get_class_weights(dataset, num_classes):
    """
    Compute class weights for handling class imbalance
    
    Args:
        dataset: Dataset object
        num_classes: Number of classes
        
    Returns:
        weights: Class weights as tensor
    """
    # Get class distribution
    class_counts = np.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Compute weights (inverse frequency)
    total_samples = len(dataset)
    weights = total_samples / (num_classes * class_counts + 1e-6)
    weights = torch.FloatTensor(weights)
    
    return weights


def print_system_info():
    """Print system and PyTorch information"""
    print("\n" + "="*70)
    print("System Information")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("="*70 + "\n")


def check_dataset_structure(data_dir):
    """
    Check if dataset directory has the correct structure
    
    Args:
        data_dir: Root directory of dataset
        
    Returns:
        valid: Whether directory structure is valid
    """
    print(f"\nChecking dataset structure in: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"✗ Directory does not exist: {data_dir}")
        return False
    
    # Check for metadata file
    metadata_paths = [
        os.path.join(data_dir, 'UrbanSound8K.csv'),
        os.path.join(data_dir, 'metadata', 'UrbanSound8K.csv'),
    ]
    
    metadata_found = False
    for path in metadata_paths:
        if os.path.exists(path):
            print(f"✓ Found metadata: {path}")
            metadata_found = True
            break
    
    if not metadata_found:
        print("✗ Metadata file (UrbanSound8K.csv) not found")
        return False
    
    # Check for fold directories
    fold_dirs = []
    for i in range(1, 11):
        fold_dir = os.path.join(data_dir, f'fold{i}')
        audio_fold_dir = os.path.join(data_dir, 'audio', f'fold{i}')
        
        if os.path.exists(fold_dir):
            fold_dirs.append(fold_dir)
        elif os.path.exists(audio_fold_dir):
            fold_dirs.append(audio_fold_dir)
    
    if len(fold_dirs) > 0:
        print(f"✓ Found {len(fold_dirs)} fold directories")
        return True
    else:
        print("✗ No fold directories found")
        return False
