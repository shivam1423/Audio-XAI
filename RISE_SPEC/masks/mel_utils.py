#!/usr/bin/env python
# coding: utf-8

import numpy as np


def mel_edges(num_mel: int, fmin: float, fmax: float) -> np.ndarray:
    """Calculate mel-frequency edges for mel-band mask generation."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mmin = hz_to_mel(fmin)
    mmax = hz_to_mel(fmax)
    mel_points = np.linspace(mmin, mmax, num_mel + 1)
    return mel_to_hz(mel_points)


def create_mel_band_mask(height: int, mel_bands: int = 64, band_keep_prob: float = 0.3) -> np.ndarray:
    """Create a mel-band mask for the given height."""
    mel_edges_array = mel_edges(num_mel=mel_bands, fmin=0.0, fmax=8000.0)
    mel_rows = np.unique(np.clip(np.round(mel_edges_array / mel_edges_array.max() * (height - 1)).astype(int), 0, height - 1))
    
    if len(mel_rows) < 2:
        mel_rows = np.array([0, height - 1])
    
    mask = np.zeros((height, mel_rows.max() + 1), dtype=np.float32)
    
    for i in range(len(mel_rows) - 1):
        y0 = int(mel_rows[i])
        y1 = int(mel_rows[i + 1]) + 1
        if np.random.rand() < band_keep_prob:
            mask[y0:y1, :] = 1.0
    
    return mask
