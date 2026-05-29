"""
UrbanSound8K Dataset implementation for Wav2Vec2 fine-tuning
"""
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import Config


class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K Dataset for urban sound classification
    
    UrbanSound8K contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
    - air_conditioner, car_horn, children_playing, dog_bark, drilling
    - engine_idling, gun_shot, jackhammer, siren, street_music
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        transform: Optional[callable] = None,
        target_sample_rate: int = 16000,
        max_duration: float = 4.0,
        train_folds: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        val_fold: int = 9,
        test_fold: int = 10
    ):
        """
        Initialize UrbanSound8K dataset
        
        Args:
            data_dir: Path to UrbanSound8K dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Optional audio transformation
            target_sample_rate: Target sample rate for audio
            max_duration: Maximum duration in seconds
            train_folds: List of fold numbers to use for training
            val_fold: Fold number to use for validation
            test_fold: Fold number to use for testing
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.train_folds = train_folds
        self.val_fold = val_fold
        self.test_fold = test_fold
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter data based on split
        self.data = self._create_split()
        
        print(f"Loaded {len(self.data)} samples for {split} split")
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load UrbanSound8K metadata from CSV file"""
        # Try multiple possible locations for metadata
        possible_paths = [
            os.path.join(self.data_dir, "metadata", "UrbanSound8K.csv"),
            os.path.join(self.data_dir, "UrbanSound8K.csv"),
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                f"UrbanSound8K metadata not found. Tried: {possible_paths}"
            )
        
        df = pd.read_csv(csv_path)
        return df
    
    def _create_split(self) -> List[Dict]:
        """Create train/val/test split based on fold information"""
        data = []
        
        for _, row in self.metadata.iterrows():
            fold = row['fold']
            class_name = row['class']
            filename = row['slice_file_name']
            class_id = row['classID']
            
            # Split based on fold numbers
            if self.split == "train" and fold in self.train_folds:
                data.append({
                    'filename': filename,
                    'class': class_name,
                    'classID': class_id,
                    'fold': fold
                })
            elif self.split == "val" and fold == self.val_fold:
                data.append({
                    'filename': filename,
                    'class': class_name,
                    'classID': class_id,
                    'fold': fold
                })
            elif self.split == "test" and fold == self.test_fold:
                data.append({
                    'filename': filename,
                    'class': class_name,
                    'classID': class_id,
                    'fold': fold
                })
        
        return data
    
    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(filepath)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to 1D
            waveform = waveform.squeeze(0)
            
            # Pad or truncate to max_duration
            max_length = int(self.max_duration * self.target_sample_rate)
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            else:
                padding = max_length - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
            
        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            # Return silence if loading fails
            return torch.zeros(int(self.max_duration * self.target_sample_rate))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get audio sample and label"""
        item = self.data[idx]
        
        # Construct file path
        # UrbanSound8K structure: audio/fold{N}/filename.wav
        filepath = os.path.join(
            self.data_dir, 
            "audio",
            f"fold{item['fold']}",
            item['filename']
        )
        
        # Load audio
        waveform = self._load_audio(filepath)
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        # Get label
        label = item['classID']
        
        return waveform, label


def create_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    target_sample_rate: int = 16000,
    max_duration: float = 4.0,
    train_folds: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    val_fold: int = 9,
    test_fold: int = 10
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_dir: Path to UrbanSound8K dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        target_sample_rate: Target sample rate
        max_duration: Maximum audio duration
        train_folds: List of folds for training
        val_fold: Fold for validation
        test_fold: Fold for testing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = UrbanSound8KDataset(
        data_dir=data_dir,
        split="train",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
        train_folds=train_folds,
        val_fold=val_fold,
        test_fold=test_fold
    )
    
    val_dataset = UrbanSound8KDataset(
        data_dir=data_dir,
        split="val",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
        train_folds=train_folds,
        val_fold=val_fold,
        test_fold=test_fold
    )
    
    test_dataset = UrbanSound8KDataset(
        data_dir=data_dir,
        split="test",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
        train_folds=train_folds,
        val_fold=val_fold,
        test_fold=test_fold
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
