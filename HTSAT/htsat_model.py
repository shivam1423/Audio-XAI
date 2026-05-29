"""
HTS-AT (Hierarchical Token Semantic Audio Transformer) Model
Based on: https://github.com/RetroCirce/HTS-Audio-Transformer

This is a simplified version for evaluation. For complete implementation,
please refer to the original repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import numpy as np


class Spectrogram_Extractor:
    """Extract mel spectrogram from waveform"""
    
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, 
                 mel_bins=64, fmin=50, fmax=14000):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            win_length=window_size,
            hop_length=hop_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=mel_bins,
            power=2.0
        )
        
    def __call__(self, waveform):
        """
        Args:
            waveform: (batch_size, samples) or (samples,)
        Returns:
            mel_spec: (batch_size, mel_bins, time_frames) or (mel_bins, time_frames)
        """
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-8)
        
        return mel_spec


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for spectrograms"""
    
    def __init__(self, img_size=(64, 1000), patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class SwinTransformerBlock(nn.Module):
    """Simplified Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Self-attention
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        
        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x


class SwinTransformerStage(nn.Module):
    """Simplified Swin Transformer Stage"""
    
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class HTSAT(nn.Module):
    """
    HTS-AT: Hierarchical Token Semantic Audio Transformer
    
    Simplified version for evaluation purposes.
    For full implementation, see: https://github.com/RetroCirce/HTS-Audio-Transformer
    """
    
    def __init__(self, num_classes=50, sample_rate=32000, window_size=1024, 
                 hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                 patch_size=4, embed_dim=96, depths=[2,2,6,2], 
                 num_heads=[4,8,16,32], window_size_spec=8):
        super().__init__()
        
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        
        # Spectrogram extractor
        self.spec_extractor = Spectrogram_Extractor(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax
        )
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=(mel_bins, 1000),  # Approximate time frames for 10s audio
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim
        )
        
        # Swin Transformer stages
        self.stages = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            dim = embed_dim * (2 ** i)
            stage = SwinTransformerStage(
                dim=dim,
                depth=depth,
                num_heads=num_head,
                window_size=window_size_spec
            )
            self.stages.append(stage)
            
            # Add patch merging (downsampling) between stages except last
            if i < len(depths) - 1:
                downsample = nn.Linear(dim, dim * 2)
                self.stages.append(downsample)
        
        # Classification head
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(final_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, samples) - raw waveform
        Returns:
            logits: (batch_size, num_classes)
        """
        # Extract spectrogram
        with torch.no_grad():
            x = self.spec_extractor(x)  # (B, mel_bins, time_frames)
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (B, 1, mel_bins, time_frames)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Transformer stages
        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.Linear):
                # Downsampling
                B, N, C = x.shape
                # Simple 2x2 pooling by averaging pairs
                if N % 2 == 0:
                    x = x.reshape(B, N // 2, 2, C).mean(dim=2)
                x = stage(x)
            else:
                # Transformer stage
                x = stage(x)
        
        # Global pooling and classification
        x = self.norm(x)  # (B, num_patches, final_dim)
        x = x.transpose(1, 2)  # (B, final_dim, num_patches)
        x = self.avgpool(x)  # (B, final_dim, 1)
        x = x.squeeze(-1)  # (B, final_dim)
        x = self.head(x)  # (B, num_classes)
        
        return x


def load_htsat_checkpoint(checkpoint_path, num_classes=50, device='cuda'):
    """
    Load HTSAT model from checkpoint
    
    Args:
        checkpoint_path: Path to .ckpt file
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        model: Loaded HTSAT model
    """
    # Create model
    model = HTSAT(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # Load state dict (with strict=False to handle missing/extra keys)
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Loaded checkpoint with strict matching")
    except Exception as e:
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint or 'acc' in checkpoint:
        acc = checkpoint.get('best_acc', checkpoint.get('acc', 'N/A'))
        print(f"Checkpoint accuracy: {acc}")
    
    return model




