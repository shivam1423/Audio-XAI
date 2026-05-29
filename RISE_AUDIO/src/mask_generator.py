#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    TIME_STRIPE_WIDTH_PX, FREQ_BAND_HEIGHT_PX, RECT_SIZE_PX,
    RECT_COUNT_RANGE, STRIPE_COUNT_RANGE, MEL_BANDS, BAND_KEEP_PROB
)
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
    mel_rows = np.unique(
        np.clip(np.round(mel_edges_array / mel_edges_array.max() * (height - 1)).astype(int), 0, height - 1))

    if len(mel_rows) < 2:
        mel_rows = np.array([0, height - 1])

    mask = np.zeros((height, mel_rows.max() + 1), dtype=np.float32)

    for i in range(len(mel_rows) - 1):
        y0 = int(mel_rows[i])
        y1 = int(mel_rows[i + 1]) + 1
        if np.random.rand() < band_keep_prob:
            mask[y0:y1, :] = 1.0

    return mask


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

class TFMaskGenerator:
    """Generates time-frequency structured masks for RISE."""

    def __init__(self, input_size, soft_masking="gaussian", edge_sigma_px=1.0):
        self.input_size = input_size
        self.soft_masking_handler = SoftMasking(soft_masking, edge_sigma_px)
        self.H, self.W = input_size

    def generate_tf_masks(
            self,
            N: int,
            time_stripe_frac: float = 0.25,
            freq_band_frac: float = 0.25,
            rect_patch_frac: float = 0.25,
            mel_band_frac: float = 0.25,
            savepath: str = 'masks_tf.npy',
            single_block: bool = False,  # new
    ) -> None:
        """Generate time-frequency structured masks."""
        print("\n\n\n single block is",single_block)
        # Adjust fractions if single mask type is selected
        if any(f == 1.0 for f in [time_stripe_frac, freq_band_frac, rect_patch_frac, mel_band_frac]):
            total = time_stripe_frac + freq_band_frac + rect_patch_frac + mel_band_frac
            time_stripe_frac /= total
            freq_band_frac /= total
            rect_patch_frac /= total
            mel_band_frac /= total

        # Determine counts for each strategy
        n_time = int(N * time_stripe_frac)
        n_freq = int(N * freq_band_frac)
        n_rect = int(N * rect_patch_frac)
        n_mel = N - n_time - n_freq - n_rect

        def rand_int(a: int, b: int) -> int:
            return int(np.random.randint(a, b + 1))

        masks = []

        # Time stripes
        for _ in range(n_time):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = 1 if single_block else rand_int(*STRIPE_COUNT_RANGE)
            for _ in range(k):
                width = rand_int(*TIME_STRIPE_WIDTH_PX)
                x0 = rand_int(0, self.W - 1)
                x1 = min(self.W, x0 + width)
                m[:, x0:x1] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Frequency bands
        for _ in range(n_freq):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = 1 if single_block else rand_int(*STRIPE_COUNT_RANGE)
            for _ in range(k):
                height = rand_int(*FREQ_BAND_HEIGHT_PX)
                y0 = rand_int(0, self.H - 1)
                y1 = min(self.H, y0 + height)
                m[y0:y1, :] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Rectangular TF patches
        for _ in range(n_rect):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = 1 if single_block else rand_int(*RECT_COUNT_RANGE)
            for _ in range(k):
                rh = rand_int(*RECT_SIZE_PX)
                rw = rand_int(*RECT_SIZE_PX)
                y0 = rand_int(0, max(0, self.H - rh))
                x0 = rand_int(0, max(0, self.W - rw))
                m[y0:y0 + rh, x0:x0 + rw] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Mel-band masks
        for _ in range(n_mel):
            if single_block:
                # exactly one contiguous mel band
                height = self.H
                rh = rand_int(*FREQ_BAND_HEIGHT_PX)
                y0 = rand_int(0, max(0, height - rh))
                y1 = y0 + rh
                m = np.zeros((height, self.W), dtype=np.float32)
                m[y0:y1, :] = 1.0
            else:
                m = create_mel_band_mask(self.H, MEL_BANDS, BAND_KEEP_PROB)
                if m.shape[1] < self.W:
                    m = np.pad(m, ((0, 0), (0, self.W - m.shape[1])), mode='constant')
                elif m.shape[1] > self.W:
                    m = m[:, :self.W]

            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = masks.reshape(-1, 1, self.H, self.W)
        np.save(savepath, masks)

        return masks
