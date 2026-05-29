"""
ESC-50 Dataset implementation for Wav2Vec2 fine-tuning
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


class ESC50Dataset(Dataset):
    """
    ESC-50 Dataset for environmental sound classification
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        transform: Optional[callable] = None,
        target_sample_rate: int = 16000,
        max_duration: float = 5.0
    ):
        """
        Initialize ESC-50 dataset
        
        Args:
            data_dir: Path to ESC-50 dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Optional audio transformation
            target_sample_rate: Target sample rate for audio
            max_duration: Maximum duration in seconds
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter data based on split
        self.data = self._create_split()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load ESC-50 metadata from CSV file"""
        csv_path = os.path.join(self.data_dir, "meta", "esc50.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ESC-50 metadata not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df
    
    def _create_split(self) -> List[Dict]:
        """Create train/val/test split based on fold information"""
        data = []
        
        for _, row in self.metadata.iterrows():
            fold = row['fold']
            category = row['category']
            filename = row['filename']
            target = row['target']
            
            # ESC-50 uses fold 1-4 for train, fold 5 for test
            # We'll use fold 4 for validation
            if self.split == "train" and fold in [1, 2, 3]:
                data.append({
                    'filename': filename,
                    'category': category,
                    'target': target,
                    'fold': fold
                })
            elif self.split == "val" and fold == 4:
                data.append({
                    'filename': filename,
                    'category': category,
                    'target': target,
                    'fold': fold
                })
            elif self.split == "test" and fold == 5:
                data.append({
                    'filename': filename,
                    'category': category,
                    'target': target,
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
        filepath = os.path.join(
            self.data_dir, 
            "audio", 
            item['filename']
        )
        
        # Load audio
        waveform = self._load_audio(filepath)
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        # Get label
        label = item['target']
        
        return waveform, label


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    target_sample_rate: int = 16000,
    max_duration: float = 5.0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_dir: Path to ESC-50 dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        target_sample_rate: Target sample rate
        max_duration: Maximum audio duration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ESC50Dataset(
        data_dir=data_dir,
        split="train",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration
    )
    
    val_dataset = ESC50Dataset(
        data_dir=data_dir,
        split="val",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration
    )
    
    test_dataset = ESC50Dataset(
        data_dir=data_dir,
        split="test",
        target_sample_rate=target_sample_rate,
        max_duration=max_duration
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

