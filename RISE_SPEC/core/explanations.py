#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masks.mask_generator import TFMaskGenerator
from masks.soft_masking import SoftMasking


class TFStructuredRISE:
    """RISE-style explainer with time–frequency structured masks and soft masking options."""

    def __init__(self, model, input_size, gpu_batch, soft_masking="gaussian", edge_sigma_px=1.0, occlusion="black"):
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.soft_masking = soft_masking
        self.edge_sigma_px = edge_sigma_px
        self.occlusion = occlusion
        self.p1 = None
        self.masks = None
        self.N = None
        
        # Initialize mask generator
        self.mask_generator = TFMaskGenerator(input_size, soft_masking, edge_sigma_px)

    def __call__(self, x):
        """Make the explainer callable like a function."""
        return self.forward(x)

    def load_masks(self, savepath):
        """Load pre-generated masks from file."""
        if not os.path.isfile(savepath):
            raise FileNotFoundError(f"Masks file not found: {savepath}")

        masks = np.load(savepath)
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = self.masks.shape[0]
        # Calculate p1 from loaded masks
        self.p1 = float(self.masks.mean().item())
        print(f"Loaded {self.N} masks from {savepath}")

    def generate_tf_masks(
        self,
        N: int,
        time_stripe_frac: float = 0.25,
        freq_band_frac: float = 0.25,
        rect_patch_frac: float = 0.25,
        mel_band_frac: float = 0.25,
        savepath: str = 'masks_tf.npy',
    ) -> None:
        """Generate time-frequency structured masks using the mask generator."""
        masks = self.mask_generator.generate_tf_masks(
            N=N,
            time_stripe_frac=time_stripe_frac,
            freq_band_frac=freq_band_frac,
            rect_patch_frac=rect_patch_frac,
            mel_band_frac=mel_band_frac,
            savepath=savepath,
        )
        
        # Load the generated masks
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = float(self.masks.mean().item())

    def forward(self, x):
        """Apply masks with configurable occlusion baseline and compute saliency."""
        N = self.N
        _, C, H, W = x.size()

        # Prepare occlusion baseline
        if self.occlusion == "black":
            baseline = torch.zeros_like(x.data)
        elif self.occlusion == "time":
            # Column-wise mean across frequencies (dim=2) for each time step
            baseline = x.data.mean(dim=2, keepdim=True).expand_as(x.data)
        elif self.occlusion == "freq":
            # Row-wise mean across time (dim=3) for each frequency bin
            baseline = x.data.mean(dim=3, keepdim=True).expand_as(x.data)
        else:
            baseline = torch.zeros_like(x.data)

        # Process masks in chunks to manage GPU memory efficiently
        chunk_size = min(self.gpu_batch, N)
        p = []
        
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_masks = self.masks[i:end_idx]
            
            # Blend input with baseline outside mask for this chunk
            # chunk_masks: [chunk_size, 1, H, W]; x/baseline: [1, C, H, W] -> broadcast to [chunk_size, C, H, W]
            chunk_stack = chunk_masks * x.data + (1.0 - chunk_masks) * baseline
            
            # Run model on this chunk
            chunk_p = self.model(chunk_stack)
            p.append(chunk_p)
            
            # Clear intermediate tensors to free memory immediately
            del chunk_masks, chunk_stack, chunk_p
            
            # Clear CUDA cache periodically to prevent fragmentation
            if i % (chunk_size * 2) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all predictions
        p = torch.cat(p)
        CL = p.size(1)
        
        # Compute saliency efficiently
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        
        # Clear final tensors
        del p
        
        return sal
