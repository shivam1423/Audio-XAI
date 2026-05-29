"""
UrbanSound8K Dataset Loader for HTSAT
Based on ESC-50 dataset loader structure
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa


class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K Dataset for HTSAT evaluation
    
    Audio is loaded at 32kHz (HTSAT's expected sample rate)
    Each clip is padded/trimmed to exactly 320000 samples (10s at 32kHz)
    
    UrbanSound8K metadata format:
    - slice_file_name: Audio filename
    - fsID: Fileset ID
    - class: Class name (e.g., 'air_conditioner')
    - fold: Fold number (1-10)
    - classID: Integer class label (0-9)
    """
    
    def __init__(self, audio_dir, metadata_path=None, target_folds=None, 
                 sample_rate=32000, clip_samples=320000):
        """
        Args:
            audio_dir: Path to audio files directory (or parent directory with fold subdirectories)
            metadata_path: Path to UrbanSound8K.csv metadata file
            target_folds: List of folds to include (e.g., [1,2,3] for training, [10] for test)
            sample_rate: Target sample rate (32000 for HTSAT)
            clip_samples: Target length in samples (320000 = 10s at 32kHz)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        
        # Find metadata file
        if metadata_path is None:
            # Search for UrbanSound8K.csv in common locations
            possible_paths = [
                os.path.join(os.path.dirname(audio_dir), 'metadata', 'UrbanSound8K.csv'),
                os.path.join(os.path.dirname(audio_dir), 'UrbanSound8K.csv'),
                os.path.join(audio_dir, 'UrbanSound8K.csv'),
                os.path.join(audio_dir, '..', 'UrbanSound8K.csv'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    metadata_path = path
                    break
            
            if metadata_path is None or not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"UrbanSound8K metadata file not found. Please provide --metadata path.\n"
                    f"Searched in: {possible_paths}"
                )
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # UrbanSound8K has fold subdirectories (fold1, fold2, ..., fold10)
        # Update audio paths to include fold directory
        if 'fold' in self.metadata.columns:
            self.metadata['audio_path'] = self.metadata.apply(
                lambda row: os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name']),
                axis=1
            )
        else:
            # If no fold column, assume all files are in audio_dir
            self.metadata['audio_path'] = self.metadata['slice_file_name'].apply(
                lambda x: os.path.join(audio_dir, x)
            )
        
        # Filter by folds if specified
        if target_folds is not None:
            if 'fold' in self.metadata.columns:
                self.metadata = self.metadata[self.metadata['fold'].isin(target_folds)]
            else:
                print(f"Warning: 'fold' column not found in metadata")
        
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples from UrbanSound8K")
        if 'fold' in self.metadata.columns:
            print(f"Folds: {sorted(self.metadata['fold'].unique())}")
        if 'classID' in self.metadata.columns:
            print(f"Classes: {len(self.metadata['classID'].unique())}")
            print(f"Class distribution:\n{self.metadata['classID'].value_counts().sort_index()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            waveform: Tensor of shape (clip_samples,) - mono audio at target sample rate
            label: Integer class label (0-9)
            filename: Original filename for reference
        """
        row = self.metadata.iloc[idx]
        
        # Get audio path
        audio_path = row['audio_path']
        
        # Verify file exists
        if not os.path.exists(audio_path):
            # Try alternative path (directly in audio_dir)
            alt_path = os.path.join(self.audio_dir, row['slice_file_name'])
            if os.path.exists(alt_path):
                audio_path = alt_path
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        try:
            # Try with torchaudio first
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to 1D tensor
            waveform = waveform.squeeze(0)
            
        except Exception as e:
            print(f"Error loading with torchaudio, trying librosa: {e}")
            # Fallback to librosa
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            waveform = torch.FloatTensor(waveform)
        
        # Pad or trim to target length
        if waveform.shape[0] < self.clip_samples:
            # Pad by repeating the audio
            num_repeats = int(np.ceil(self.clip_samples / waveform.shape[0]))
            waveform = waveform.repeat(num_repeats)
        
        # Trim to exact length
        waveform = waveform[:self.clip_samples]
        
        # Get label (classID is 0-9)
        if 'classID' in row:
            label = int(row['classID'])
        elif 'class' in row:
            # Map class name to ID if needed
            class_name = row['class']
            class_mapping = {
                'air_conditioner': 0, 'car_horn': 1, 'children_playing': 2,
                'dog_bark': 3, 'drilling': 4, 'engine_idling': 5,
                'gun_shot': 6, 'jackhammer': 7, 'siren': 8, 'street_music': 9
            }
            label = class_mapping.get(class_name, 0)
        else:
            raise ValueError("No class label found in metadata")
        
        filename = row['slice_file_name']
        
        return waveform, label, filename


def get_dataloader(audio_dir, metadata_path=None, target_folds=None, 
                   batch_size=32, num_workers=4, sample_rate=32000, 
                   clip_samples=320000, shuffle=False):
    """
    Create a DataLoader for UrbanSound8K
    
    Args:
        audio_dir: Path to audio files (parent directory containing fold1, fold2, etc.)
        metadata_path: Path to UrbanSound8K.csv (optional, will search if not provided)
        target_folds: List of folds to include (e.g., [1,2,3] or [10] for test)
        batch_size: Batch size
        num_workers: Number of worker processes
        sample_rate: Target sample rate
        clip_samples: Target clip length in samples
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance and dataset
    """
    dataset = UrbanSound8KDataset(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        target_folds=target_folds,
        sample_rate=sample_rate,
        clip_samples=clip_samples
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset

