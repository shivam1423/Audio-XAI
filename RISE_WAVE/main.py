#!/usr/bin/env python
# coding: utf-8

"""Main script for TF-Structured RISE: Orchestrates all modular components."""

import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder

# Import modular components
from config.cli_parser import parse_cli_args
from config.settings import (
    DEFAULT_WORKERS, MODEL_INPUT_SIZE, DEFAULT_DATADIR, DEFAULT_AUDIO_DIR, DEFAULT_GPU_BATCH,
    MASKS_DIR, SALIENCY_DIR, IMG_EXTENSIONS
)
from config.model_configs import get_model_config, is_spectrogram_model
from core.models import setup_model  # Keep for backward compatibility
from core.model_factory import create_model, get_preprocessor
from core.datasets import CustomImageDataset, create_data_loader
from core.audio_datasets import AudioDataset, create_audio_data_loader
from core.explanations import TFStructuredRISE
from saliency.explainer import explain_all, save_saliency_results
from saliency.visualization import save_individual_saliency_maps
from rise_utils import preprocess, RangeSampler

# Setup CUDA
cudnn.benchmark = True

# Add support for uppercase image extensions
folder.IMG_EXTENSIONS = IMG_EXTENSIONS


def setup_arguments(cli_args, config):
    """Setup arguments object from CLI arguments."""
    class Dummy:
        pass
    
    args = Dummy()
    args.workers = DEFAULT_WORKERS
    args.datadir = cli_args.datadir
    args.audio_dir = cli_args.audio_dir
    args.range = None
    args.input_size = config['input_size']
    args.gpu_batch = cli_args.gpu_batch
    args.model_type = cli_args.model_type
    
    return args


def determine_mask_fractions(cli_args):
    """Determine mask fractions based on mask type."""
    if cli_args.mask_type == 'time':
        fractions = dict(time=1.0, freq=0.0, rect=0.0, mel=0.0)
    elif cli_args.mask_type == 'freq':
        fractions = dict(time=0.0, freq=1.0, rect=0.0, mel=0.0)
    elif cli_args.mask_type == 'rect':
        fractions = dict(time=0.0, freq=0.0, rect=1.0, mel=0.0)
    elif cli_args.mask_type == 'mel':
        fractions = dict(time=0.0, freq=0.0, rect=0.0, mel=1.0)
    else:  # 'all'
        fractions = dict(
            time=cli_args.time_stripe_frac,
            freq=cli_args.freq_band_frac,
            rect=cli_args.rect_patch_frac,
            mel=cli_args.mel_band_frac
        )
    
    return fractions


def setup_output_paths(cli_args):
    """Setup output paths for masks and results."""
    masking_style = f"_{cli_args.soft_masking}" if cli_args.soft_masking != "none" else "_discrete"
    mask_suffix = f"_{cli_args.mask_type}" if cli_args.mask_type != "all" else "_combined"
    occlusion_suffix = f"_occlusion_{cli_args.occlusion}"
    
    # Use custom mask name if provided, otherwise use auto-generated
    if cli_args.mask_name:
        maskspath = os.path.join(MASKS_DIR, f'{cli_args.mask_name}.npy')
    else:
        maskspath = os.path.join(MASKS_DIR, f'masks{masking_style}{mask_suffix}.npy')

    # Use custom output name if provided, otherwise use auto-generated
    if cli_args.output_name:
        output_dir = os.path.join(SALIENCY_DIR, cli_args.output_name)
    else:
        output_dir = os.path.join(SALIENCY_DIR, f'saliency_maps{masking_style}{mask_suffix}{occlusion_suffix}')
    
    return maskspath, output_dir, masking_style, mask_suffix, occlusion_suffix


def main():
    """Main execution function."""
    # Parse CLI arguments
    cli = parse_cli_args()
    
    # Get model configuration
    config = get_model_config(cli.model_type)
    
    # Setup arguments
    args = setup_arguments(cli, config)
    
    # Determine whether to use audio files or images
    USE_AUDIO = is_spectrogram_model(cli.model_type)  # Use audio for all models
    
    # Create dataset and data loader
    if USE_AUDIO:
        dataset = AudioDataset(args.audio_dir)
        if args.range is not None:
            data_loader = create_audio_data_loader(dataset, batch_size=1, num_workers=args.workers,
                                                  sampler=RangeSampler(args.range))
        else:
            data_loader = create_audio_data_loader(dataset, batch_size=1, num_workers=args.workers)
        # Get preprocessor
        preprocessor = get_preprocessor(cli.model_type, config)
    else:
        # Backward compatibility: use pre-generated spectrograms
        dataset = CustomImageDataset(args.datadir, preprocess)
        if args.range is not None:
            data_loader = create_data_loader(dataset, args, RangeSampler(args.range))
        else:
            data_loader = create_data_loader(dataset, args)
        preprocessor = None
    
    # Setup model using factory
    model = create_model(cli.model_type, weights_path=cli.model_weights, num_classes=50)
    
    # Setup explainer
    explainer = TFStructuredRISE(
        model,
        args.input_size,
        args.gpu_batch,
        soft_masking=cli.soft_masking,
        edge_sigma_px=cli.edge_sigma_px,
        occlusion=cli.occlusion
    )
    
    # Determine mask fractions
    fractions = determine_mask_fractions(cli)
    
    # Setup output paths
    maskspath, output_dir, masking_style, mask_suffix, occlusion_suffix = setup_output_paths(cli)
    
    # Ensure directories exist
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(SALIENCY_DIR, exist_ok=True)
    
    # Generate or load masks
    if cli.generate_new or not os.path.isfile(maskspath):
        explainer.generate_tf_masks(
            N=cli.N,
            time_stripe_frac=fractions['time'],
            freq_band_frac=fractions['freq'],
            rect_patch_frac=fractions['rect'],
            mel_band_frac=fractions['mel'],
            savepath=maskspath,
        )
    else:
        explainer.load_masks(maskspath)
        print(f'Masks loaded from {maskspath}')
    
    # Generate explanations
    if USE_AUDIO:
        explanations, targets, filenames = explain_all(data_loader, explainer, model, preprocessor)
    else:
        explanations, targets, filenames = explain_all(data_loader, explainer, model)
    
    # Save results
    results_info = {
        'input_size': args.input_size,
        'mask_type': f'tf_structured{mask_suffix}',
        'soft_masking': cli.soft_masking,
        'edge_sigma_px': cli.edge_sigma_px,
        'occlusion': cli.occlusion,
        'mask_fractions': fractions,
        'N': cli.N,
        'mask_file': maskspath,
        'occlusion_method': cli.occlusion
    }
    
    save_saliency_results(explanations, targets, filenames, output_dir, results_info)
    
    # Save individual files
    ref_dir = args.audio_dir if USE_AUDIO else args.datadir
    if USE_AUDIO:
        save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, ref_dir, model, preprocessor)
    else:
        save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, ref_dir, model)


if __name__ == "__main__":
    main()
