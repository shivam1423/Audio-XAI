"""
ESC-50 Dataset Loader for HTSAT
Based on: https://github.com/RetroCirce/HTS-Audio-Transformer
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa


class ESC50Dataset(Dataset):
    """
    ESC-50 Dataset for HTSAT evaluation
    
    Audio is loaded at 32kHz (HTSAT's expected sample rate)
    Each clip is 5 seconds, padded/trimmed to exactly 320000 samples (10s at 32kHz)
    """
    
    def __init__(self, audio_dir, metadata_path=None, target_folds=None, 
                 sample_rate=32000, clip_samples=320000):
        """
        Args:
            audio_dir: Path to audio files directory
            metadata_path: Path to esc50.csv (optional, will search if not provided)
            target_folds: List of folds to include (e.g., [1,3,4,5] for training)
            sample_rate: Target sample rate (32000 for HTSAT)
            clip_samples: Target length in samples (320000 = 10s at 32kHz)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        
        # Find metadata file
        if metadata_path is None:
            # Search for esc50.csv in common locations
            possible_paths = [
                os.path.join(os.path.dirname(audio_dir), 'meta', 'esc50.csv'),
                os.path.join(os.path.dirname(audio_dir), 'esc50.csv'),
                os.path.join(audio_dir, 'esc50.csv'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    metadata_path = path
                    break
            
            # If still not found, create from audio files
            if metadata_path is None or not os.path.exists(metadata_path):
                print(f"Warning: Metadata file not found. Creating from audio files...")
                self.metadata = self._create_metadata_from_files()
            else:
                self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.read_csv(metadata_path)
        
        # Filter by folds if specified
        if target_folds is not None:
            if 'fold' in self.metadata.columns:
                self.metadata = self.metadata[self.metadata['fold'].isin(target_folds)]
            else:
                print(f"Warning: 'fold' column not found in metadata")
        
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples from ESC-50")
        if 'fold' in self.metadata.columns:
            print(f"Folds: {sorted(self.metadata['fold'].unique())}")
        if 'target' in self.metadata.columns:
            print(f"Classes: {len(self.metadata['target'].unique())}")
    
    def _create_metadata_from_files(self):
        """
        Create metadata DataFrame from audio files
        ESC-50 filename format: {fold}-{clip_id}-{take}-{target}.wav
        Example: 1-100032-A-0.wav
        """
        files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        
        data = []
        for filename in files:
            try:
                parts = filename.replace('.wav', '').split('-')
                if len(parts) >= 4:
                    fold = int(parts[0])
                    clip_id = parts[1]
                    take = parts[2]
                    target = int(parts[3])
                    
                    data.append({
                        'filename': filename,
                        'fold': fold,
                        'target': target,
                        'category': f'class_{target}',
                        'esc10': False,
                        'src_file': clip_id,
                        'take': take
                    })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse filename {filename}: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            waveform: Tensor of shape (clip_samples,) - mono audio at target sample rate
            label: Integer class label
            filename: Original filename for reference
        """
        row = self.metadata.iloc[idx]
        
        # Get filename
        if 'filename' in row:
            filename = row['filename']
        else:
            # Reconstruct filename from metadata
            filename = f"{row['fold']}-{row['src_file']}-{row['take']}-{row['target']}.wav"
        
        audio_path = os.path.join(self.audio_dir, filename)
        
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
        
        # Get label
        label = int(row['target'])
        
        return waveform, label, filename


def get_dataloader(audio_dir, metadata_path=None, target_folds=None, 
                   batch_size=32, num_workers=4, sample_rate=32000, 
                   clip_samples=320000, shuffle=False):
    """
    Create a DataLoader for ESC-50
    
    Args:
        audio_dir: Path to audio files
        metadata_path: Path to esc50.csv (optional)
        target_folds: List of folds to include
        batch_size: Batch size
        num_workers: Number of worker processes
        sample_rate: Target sample rate
        clip_samples: Target clip length in samples
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = ESC50Dataset(
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




