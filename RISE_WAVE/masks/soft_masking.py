#!/usr/bin/env python
# coding: utf-8

import numpy as np


class SoftMasking:
    """Handles different soft masking strategies for masks."""
    
    def __init__(self, method="gaussian", edge_sigma_px=1.0):
        self.method = method
        self.edge_sigma_px = edge_sigma_px
    
    def apply(self, mask2d: np.ndarray) -> np.ndarray:
        """Apply the selected soft masking method."""
        if self.method == "gaussian":
            return self._gaussian_soften(mask2d)
        elif self.method == "bilinear":
            return self._bilinear_upsample(mask2d)
        else:
            return mask2d.astype(np.float32)
    
    def _gaussian_soften(self, mask2d: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for edge softening."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(mask2d.astype(np.float32), sigma=self.edge_sigma_px)
        except ImportError:
            print("Warning: scipy not available, using hard masks")
            return mask2d.astype(np.float32)
    
    def _bilinear_upsample(self, mask2d: np.ndarray) -> np.ndarray:
        """Apply bilinear upsampling like original RISE."""
        try:
            from skimage.transform import resize
        except ImportError:
            print("Warning: skimage not available, using hard masks")
            return mask2d.astype(np.float32)
        
        H, W = mask2d.shape
        
        # Original RISE approach: coarse grid -> bilinear upsample
        s = 8  # Grid downsampling factor
        
        # Step 1: Downsample to coarse grid
        coarse_h, coarse_w = max(1, H // s), max(1, W // s)
        coarse_mask = resize(mask2d, (coarse_h, coarse_w), order=1, mode='reflect', anti_aliasing=True)
        
        # Step 2: Upsample with bilinear interpolation (like original RISE)
        smooth_mask = resize(coarse_mask, (H, W), order=1, mode='reflect', anti_aliasing=False)
        
        return smooth_mask.astype(np.float32)
