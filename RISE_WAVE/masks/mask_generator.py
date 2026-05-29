#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .soft_masking import SoftMasking
from .mel_utils import create_mel_band_mask
from config.settings import (
    TIME_STRIPE_WIDTH_PX, FREQ_BAND_HEIGHT_PX, RECT_SIZE_PX,
    RECT_COUNT_RANGE, STRIPE_COUNT_RANGE, MEL_BANDS, BAND_KEEP_PROB
)


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
    ) -> None:
        """Generate time-frequency structured masks."""
        
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

        # Time stripes: choose several vertical stripes
        for _ in range(n_time):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = rand_int(*STRIPE_COUNT_RANGE)
            for _ in range(k):
                width = rand_int(*TIME_STRIPE_WIDTH_PX)
                x0 = rand_int(0, self.W - 1)
                x1 = min(self.W, x0 + width)
                m[:, x0:x1] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Frequency bands: choose several horizontal bands
        for _ in range(n_freq):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = rand_int(*STRIPE_COUNT_RANGE)
            for _ in range(k):
                height = rand_int(*FREQ_BAND_HEIGHT_PX)
                y0 = rand_int(0, self.H - 1)
                y1 = min(self.H, y0 + height)
                m[y0:y1, :] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Rectangular TF patches: several rectangles aligned to axes
        for _ in range(n_rect):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            k = rand_int(*RECT_COUNT_RANGE)
            for _ in range(k):
                rh = rand_int(*RECT_SIZE_PX)
                rw = rand_int(*RECT_SIZE_PX)
                y0 = rand_int(0, max(0, self.H - rh))
                x0 = rand_int(0, max(0, self.W - rw))
                m[y0:y0 + rh, x0:x0 + rw] = 1.0
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        # Mel-band masks: select mel bands to keep
        for _ in range(n_mel):
            m = create_mel_band_mask(self.H, MEL_BANDS, BAND_KEEP_PROB)
            # Ensure mask has correct width
            if m.shape[1] < self.W:
                m = np.pad(m, ((0, 0), (0, self.W - m.shape[1])), mode='constant')
            elif m.shape[1] > self.W:
                m = m[:, :self.W]
            
            if self.soft_masking_handler.method != "none":
                m = self.soft_masking_handler.apply(m)
            masks.append(m)

        masks = np.stack(masks, axis=0)  # [N, H, W]
        masks = masks.reshape(-1, 1, self.H, self.W)
        np.save(savepath, masks)
        
        return masks
