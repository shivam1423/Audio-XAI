#!/usr/bin/env python
# coding: utf-8

import argparse


def parse_cli_args():
    """Parse command line arguments for TF-Structured RISE."""
    parser = argparse.ArgumentParser(description="Generate saliency maps with TF-structured masks.")

    # Mask type selection
    parser.add_argument(
        "--mask_type",
        type=str,
        default="all",
        choices=["all", "time", "freq", "rect", "mel"],
        help="Which mask strategy to use: all (combined), time (vertical stripes), freq (horizontal bands), rect (rectangular patches), mel (mel-frequency bands)."
    )

    # Mask generation parameters
    parser.add_argument("--N", type=int, default=6000, help="Number of masks to generate (default: 6000)")
    parser.add_argument("--time_stripe_frac", type=float, default=0.25,
                        help="Fraction of time stripe masks (default: 0.25)")
    parser.add_argument("--freq_band_frac", type=float, default=0.25,
                        help="Fraction of frequency band masks (default: 0.25)")
    parser.add_argument("--rect_patch_frac", type=float, default=0.25,
                        help="Fraction of rectangular patch masks (default: 0.25)")
    parser.add_argument("--mel_band_frac", type=float, default=0.25, help="Fraction of mel-band masks (default: 0.25)")

    # Soft masking options
    parser.add_argument(
        "--soft_masking",
        type=str,
        default="bilinear",
        choices=["gaussian", "bilinear", "none"],
        help="Soft masking method: gaussian (Gaussian blur), bilinear (bilinear upsampling like RISE), none (hard masks)"
    )
    parser.add_argument("--edge_sigma_px", type=float, default=1.0,
                        help="Gaussian blur sigma for edge softening (default: 1.0)")

    # Occlusion baseline options
    parser.add_argument(
        "--occlusion",
        type=str,
        default="black",
        choices=["black", "time", "freq"],
        help="Occlusion baseline for masked_audio-out regions: black (zeros), time (column-wise mean across frequencies), freq (row-wise mean across time)"
    )

    # General parameters
    parser.add_argument("--datadir", type=str, default='ESC50_spectrograms/',
                        help="Dataset root directory (default: ESC50_spectrograms/)")
    parser.add_argument("--audio_dir", type=str, default='test_audio/',
                        help="Raw audio directory (default: test_audio/) - use relative path for cluster")
    parser.add_argument("--gpu_batch", type=int, default=250, help="Batch size for mask application (default: 250)")
    parser.add_argument("--generate_new", action="store_true", help="Force generation of new masks even if file exists")

    # Model selection
    parser.add_argument("--model_type", type=str, default='resnet50',
                        choices=['resnet50', 'htsat', 'wav2vec2'],
                        help="Model architecture to use (default: resnet50)")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Path to model weights file (default: uses config from model_configs.py)")

    # Custom naming options
    parser.add_argument("--output_name", type=str, default=None,
                        help="Custom name for output folder (default: auto-generated)")
    parser.add_argument("--mask_name", type=str, default=None,
                        help="Custom name for mask file (default: auto-generated)")

    return parser.parse_args()
