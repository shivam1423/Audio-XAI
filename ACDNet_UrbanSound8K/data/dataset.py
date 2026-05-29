"""
UrbanSound8K Dataset Loader for ACDNet
Handles fold-based splitting and returns raw audio waveforms
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .preprocessor import AudioPreprocessor


class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K Dataset for ACDNet training and evaluation
    
    Returns raw audio waveforms at 20kHz, 1.5 seconds
    Compatible with BC Learning data generator
    
    UrbanSound8K structure:
    - 10 classes of urban sounds
    - 10 folds for cross-validation
    - Audio files organized in fold1/, fold2/, ..., fold10/ subdirectories
    - Metadata in UrbanSound8K.csv
    """
    
    def __init__(self, data_dir, folds, sr=20000, input_length=30000, metadata_file='UrbanSound8K.csv'):
        """
        Initialize UrbanSound8K dataset
        
        Args:
            data_dir: Root directory containing UrbanSound8K dataset
            folds: List of fold numbers to include (e.g., [1,2,3,4,5,6,7,8] for training)
            sr: Target sampling rate (20kHz for ACDNet)
            input_length: Target audio length in samples (30000 = 1.5s at 20kHz)
            metadata_file: Name of metadata CSV file
        """
        self.data_dir = data_dir
        self.folds = folds
        self.sr = sr
        self.input_length = input_length
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(target_sr=sr, target_length=input_length)
        
        # Find and load metadata
        self.metadata_path = self._find_metadata(metadata_file)
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Filter by folds
        if 'fold' not in self.metadata.columns:
            raise ValueError("Metadata file must contain 'fold' column")
        
        self.metadata = self.metadata[self.metadata['fold'].isin(folds)]
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Build file paths
        self._build_file_paths()
        
        # Extract class information
        self.num_classes = len(self.metadata['classID'].unique())
        
        print(f"Loaded UrbanSound8K dataset:")
        print(f"  - Folds: {sorted(self.folds)}")
        print(f"  - Total samples: {len(self.metadata)}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Sample rate: {self.sr} Hz")
        print(f"  - Input length: {self.input_length} samples ({self.input_length/self.sr:.1f}s)")
    
    def _find_metadata(self, metadata_file):
        """Find metadata CSV file in common locations"""
        possible_paths = [
            os.path.join(self.data_dir, metadata_file),
            os.path.join(self.data_dir, 'metadata', metadata_file),
            os.path.join(self.data_dir, '..', metadata_file),
            os.path.join(os.path.dirname(self.data_dir), metadata_file),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Metadata file '{metadata_file}' not found. Searched in:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def _build_file_paths(self):
        """Build full file paths for audio files"""
        # Try different possible structures for UrbanSound8K
        
        # Structure 1: data_dir/fold1/, fold2/, etc.
        self.metadata['audio_path'] = self.metadata.apply(
            lambda row: os.path.join(
                self.data_dir, 
                f"fold{row['fold']}", 
                row['slice_file_name']
            ),
            axis=1
        )
        
        # Verify that at least some files exist
        sample_paths = self.metadata['audio_path'].head(5)
        existing = sum(1 for p in sample_paths if os.path.exists(p))
        
        if existing == 0:
            # Structure 2: data_dir/audio/fold1/, fold2/, etc. (standard UrbanSound8K)
            self.metadata['audio_path'] = self.metadata.apply(
                lambda row: os.path.join(
                    self.data_dir,
                    'audio',
                    f"fold{row['fold']}", 
                    row['slice_file_name']
                ),
                axis=1
            )
            sample_paths = self.metadata['audio_path'].head(5)
            existing = sum(1 for p in sample_paths if os.path.exists(p))
            
            if existing == 0:
                # Structure 3: audio files directly in data_dir
                self.metadata['audio_path'] = self.metadata['slice_file_name'].apply(
                    lambda x: os.path.join(self.data_dir, x)
                )
                sample_paths = self.metadata['audio_path'].head(5)
                existing = sum(1 for p in sample_paths if os.path.exists(p))
                
                if existing == 0:
                    raise FileNotFoundError(
                        f"Audio files not found. Please check data_dir: {self.data_dir}\n"
                        f"Expected structures:\n"
                        f"  - {self.data_dir}/fold1/, fold2/, etc.\n"
                        f"  - {self.data_dir}/audio/fold1/, fold2/, etc. (standard)\n"
                        f"  - {self.data_dir}/<audio_files>"
                    )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            sound: Preprocessed audio waveform (numpy array, 1D)
            label: Integer class label (0-9)
            
        Note: For BC Learning, the generator will mix two samples
        """
        row = self.metadata.iloc[idx]
        
        # Get audio path and label
        audio_path = row['audio_path']
        label = int(row['classID'])
        
        # Verify file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        try:
            sound = self.preprocessor.preprocess(audio_path)
        except Exception as e:
            raise Exception(f"Error preprocessing {audio_path}: {e}")
        
        return sound, label
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        return self.metadata['classID'].value_counts().sort_index()
    
    def get_samples_by_class(self, class_id):
        """Get all samples for a specific class"""
        class_metadata = self.metadata[self.metadata['classID'] == class_id]
        return class_metadata.index.tolist()


def create_data_loaders(config, num_workers=4):
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        config: ACDNetConfig object with dataset parameters
        num_workers: Number of workers for data loading
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Training dataset
    train_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=config.train_folds,
        sr=config.sr,
        input_length=config.input_length
    )
    
    # Validation dataset
    val_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=[config.val_fold],
        sr=config.sr,
        input_length=config.input_length
    )
    
    # Test dataset
    test_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=[config.test_fold],
        sr=config.sr,
        input_length=config.input_length
    )
    
    return train_dataset, val_dataset, test_dataset


def get_data_info(config):
    """
    Get information about the dataset splits
    
    Args:
        config: ACDNetConfig object
    
    Returns:
        Dictionary with dataset statistics
    """
    train_ds, val_ds, test_ds = create_data_loaders(config)
    
    info = {
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'total_samples': len(train_ds) + len(val_ds) + len(test_ds),
        'num_classes': train_ds.num_classes,
        'train_folds': config.train_folds,
        'val_fold': config.val_fold,
        'test_fold': config.test_fold,
        'sample_rate': config.sr,
        'input_length': config.input_length
    }
    
    return info
