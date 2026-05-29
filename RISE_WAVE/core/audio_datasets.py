#!/usr/bin/env python
# coding: utf-8

"""Audio dataset classes for loading raw audio files."""

import os
import glob
import torch
import torch.utils.data
from typing import Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import AUDIO_EXTENSIONS


class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for loading raw audio files (for spectrogram models).
    Returns filepath instead of loading audio, allowing preprocessing to happen later.
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize audio dataset.
        
        Args:
            root_dir: Root directory containing audio files
            transform: Optional transform (not used, kept for compatibility)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.audio_files = []
        
        print(f"Searching for audio files in: {os.path.abspath(root_dir)}")
        
        # Find all audio files recursively
        for ext in AUDIO_EXTENSIONS:
            pattern = os.path.join(root_dir, '**', f'*{ext}')
            self.audio_files.extend(glob.glob(pattern, recursive=True))
        
        if len(self.audio_files) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx) -> Tuple[str, int, str]:
        """
        Get item at index.
        
        Returns:
            filepath: Path to audio file
            label: Dummy label (0)
            filepath: Same path for reference
        """
        filepath = self.audio_files[idx]
        return filepath, 0, filepath


class WaveformDataset(torch.utils.data.Dataset):
    """
    Dataset for loading raw audio files (for waveform models like Wav2Vec2).
    Returns filepath instead of loading audio, allowing preprocessing to happen later.
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize waveform dataset.
        
        Args:
            root_dir: Root directory containing audio files
            transform: Optional transform (not used, kept for compatibility)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.audio_files = []
        
        print(f"Searching for audio files in: {os.path.abspath(root_dir)}")
        
        # Find all audio files recursively
        for ext in AUDIO_EXTENSIONS:
            pattern = os.path.join(root_dir, '**', f'*{ext}')
            self.audio_files.extend(glob.glob(pattern, recursive=True))
        
        if len(self.audio_files) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx) -> Tuple[str, int, str]:
        """
        Get item at index.
        
        Returns:
            filepath: Path to audio file
            label: Dummy label (0)
            filepath: Same path for reference
        """
        filepath = self.audio_files[idx]
        return filepath, 0, filepath


def create_audio_data_loader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 2,
    shuffle: bool = False,
    pin_memory: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None
):
    """
    Create data loader for audio dataset.
    
    Args:
        dataset: AudioDataset or WaveformDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        sampler: Optional sampler
        
    Returns:
        DataLoader instance
    """
    if sampler is not None:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Can't shuffle with sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return data_loader

