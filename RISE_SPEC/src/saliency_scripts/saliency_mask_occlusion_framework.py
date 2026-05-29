#!/usr/bin/env python
# coding: utf-8

# Time–Frequency Structured RISE: Combined script with CLI and soft masking options

import os
import math
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# Data setup
import glob
from PIL import Image
from src.core.saliency_utils import *
from src.core.explanations import RISE

# Import model factory and audio datasets
from core.model_factory import create_model, get_preprocessor
from core.audio_datasets import AudioDataset, create_audio_data_loader
from config.model_configs import get_model_config, get_dataset_config, get_model_config_for_dataset

cudnn.benchmark = True

# Add support for uppercase image extensions
import torchvision.datasets.folder as folder

folder.IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF', '.TIFF'
)



def parse_cli_args():
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
    parser.add_argument("--dataset", type=str, default='urbansound8k',
                        choices=['esc50', 'urbansound8k'],
                        help="Dataset to use (default: urbansound8k)")
    parser.add_argument("--datadir", type=str, default=None,
                        help="Dataset root directory (overrides dataset default)")
    parser.add_argument("--gpu_batch", type=int, default=150, help="Batch size for mask application (default: 250)")
    parser.add_argument("--generate_new", action="store_true", help="Force generation of new masks even if file exists")

    # Model selection
    parser.add_argument("--model_type", type=str, default='resnet50',
                        choices=['resnet50', 'htsat'],
                        help="Model architecture to use (default: resnet50)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Raw audio directory (overrides dataset default)")
    parser.add_argument("--use_audio", action="store_true",
                        help="Use raw audio files instead of pre-generated spectrograms")
    parser.add_argument("--htsat_waveform_input", action="store_true",
                        help="Force HTSAT to use waveform input (default: spectrogram preprocessing for saliency)")

    # Custom naming options
    parser.add_argument("--output_name", type=str, default=None,
                        help="Custom name for output folder (default: auto-generated)")
    parser.add_argument("--mask_name", type=str, default=None,
                        help="Custom name for mask file (default: auto-generated)")

    return parser.parse_args()


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                    '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']:
            self.image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0, img_path


class TFStructuredRISE(RISE):
    """RISE-style explainer with time–frequency structured masks and soft masking options."""

    def __init__(self, model, input_size, gpu_batch, soft_masking="gaussian", edge_sigma_px=1.0, occlusion="black"):
        super().__init__(model, input_size, gpu_batch)
        self.soft_masking = soft_masking
        self.edge_sigma_px = edge_sigma_px
        self.occlusion = occlusion
        self.p1 = None

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
            time_stripe_width_px: tuple = (4, 24),
            freq_band_height_px: tuple = (4, 24),
            rect_size_px: tuple = (8, 48),
            rect_count_range: tuple = (1, 6),
            stripe_count_range: tuple = (1, 12),
            mel_bands: int = 64,
            band_keep_prob: float = 0.3,
            savepath: str = 'masks_tf.npy',
    ) -> None:
        H, W = self.input_size

        # Adjust fractions if single mask type is selected
        if any(f == 1.0 for f in [time_stripe_frac, freq_band_frac, rect_patch_frac, mel_band_frac]):
            # Single mask type selected, normalize to 1.0
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
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*stripe_count_range)
            for _ in range(k):
                width = rand_int(*time_stripe_width_px)
                x0 = rand_int(0, W - 1)
                x1 = min(W, x0 + width)
                m[:, x0:x1] = 1.0
            if self.soft_masking != "none":
                m = self._apply_soft_masking(m)
            masks.append(m)

        # Frequency bands: choose several horizontal bands
        for _ in range(n_freq):
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*stripe_count_range)
            for _ in range(k):
                height = rand_int(*freq_band_height_px)
                y0 = rand_int(0, H - 1)
                y1 = min(H, y0 + height)
                m[y0:y1, :] = 1.0
            if self.soft_masking != "none":
                m = self._apply_soft_masking(m)
            masks.append(m)

        # Rectangular TF patches: several rectangles aligned to axes
        for _ in range(n_rect):
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*rect_count_range)
            for _ in range(k):
                rh = rand_int(*rect_size_px)
                rw = rand_int(*rect_size_px)
                y0 = rand_int(0, max(0, H - rh))
                x0 = rand_int(0, max(0, W - rw))
                m[y0:y0 + rh, x0:x0 + rw] = 1.0
            if self.soft_masking != "none":
                m = self._apply_soft_masking(m)
            masks.append(m)

        # Mel-band masks: select mel bands to keep
        mel_edges = self._mel_edges(num_mel=mel_bands, fmin=0.0, fmax=8000.0)
        mel_rows = np.unique(np.clip(np.round(mel_edges / mel_edges.max() * (H - 1)).astype(int), 0, H - 1))
        if len(mel_rows) < 2:
            mel_rows = np.array([0, H - 1])

        for _ in range(n_mel):
            m = np.zeros((H, W), dtype=np.float32)
            for i in range(len(mel_rows) - 1):
                y0 = int(mel_rows[i])
                y1 = int(mel_rows[i + 1]) + 1
                if np.random.rand() < band_keep_prob:
                    m[y0:y1, :] = 1.0
            if self.soft_masking != "none":
                m = self._apply_soft_masking(m)
            masks.append(m)

        masks = np.stack(masks, axis=0)  # [N, H, W]
        masks = masks.reshape(-1, 1, H, W)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.save(savepath, masks)
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

        # Blend input with baseline outside mask
        # self.masks: [N, 1, H, W]; x/baseline: [1, C, H, W] -> broadcast to [N, C, H, W]
        stack = self.masks * x.data + (1.0 - self.masks) * baseline

        # Get baseline prediction for saliency computation
        baseline_pred = None
        try:
            baseline_input = baseline.unsqueeze(0) if baseline.dim() == 3 else baseline
            baseline_pred = self.model(baseline_input)
        except Exception:
            pass
        
        # Ensure model is in eval mode for stable batch normalization
        was_training = self.model.training
        self.model.eval()
        
        # Run model in batches along N
        p = []
        with torch.no_grad():  # No gradients needed for RISE
            for i in range(0, N, self.gpu_batch):
                batch_stack = stack[i:min(i + self.gpu_batch, N)]
                batch_pred = self.model(batch_stack)
                p.append(batch_pred)
        
        # Restore training mode if it was training
        if was_training:
            self.model.train()
        p = torch.cat(p)
        CL = p.size(1)
        
        # Compute saliency using baseline-subtracted predictions
        # This is critical when baseline predicts a different class than original
        # Standard RISE uses absolute predictions, but baseline subtraction gives true contribution
        if baseline_pred is not None:
            # Subtract baseline prediction to get contribution relative to baseline
            # This ensures we measure how pixels help/hurt relative to the baseline, not absolute values
            p_diff = p - baseline_pred.expand_as(p)
            sal = torch.matmul(p_diff.data.transpose(0, 1), self.masks.view(N, H * W))
        else:
            # Fallback: use absolute predictions if baseline not available
            sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        
        return sal

    def _apply_soft_masking(self, mask2d: np.ndarray) -> np.ndarray:
        """Apply the selected soft masking method."""
        if self.soft_masking == "gaussian":
            return self._gaussian_soften(mask2d)
        elif self.soft_masking == "bilinear":
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
        from skimage.transform import resize

        H, W = mask2d.shape

        # Original RISE approach: coarse grid -> bilinear upsample
        s = 8  # Grid downsampling factor

        # Step 1: Downsample to coarse grid
        coarse_h, coarse_w = max(1, H // s), max(1, W // s)
        coarse_mask = resize(mask2d, (coarse_h, coarse_w), order=1, mode='reflect', anti_aliasing=True)

        # Step 2: Upsample with bilinear interpolation (like original RISE)
        smooth_mask = resize(coarse_mask, (H, W), order=1, mode='reflect', anti_aliasing=False)

        return smooth_mask.astype(np.float32)

    @staticmethod
    def _mel_edges(num_mel: int, fmin: float, fmax: float) -> np.ndarray:
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mmin = hz_to_mel(fmin)
        mmax = hz_to_mel(fmax)
        mel_points = np.linspace(mmin, mmax, num_mel + 1)
        return mel_to_hz(mel_points)


def main():
    cli = parse_cli_args()

    # Get dataset + model configuration
    dataset_config = get_dataset_config(cli.dataset)
    config = get_model_config_for_dataset(cli.model_type, cli.dataset)

    # Setup arguments
    args = Dummy()
    args.workers   = 2
    args.datadir   = cli.datadir   or dataset_config['default_spectrogram_dir']
    args.audio_dir = cli.audio_dir or dataset_config['default_audio_dir']
    args.range = None
    args.input_size = config['input_size']
    args.gpu_batch = cli.gpu_batch
    args.model_type = cli.model_type

    # Determine whether to use audio files
    USE_AUDIO = cli.use_audio or cli.model_type != 'resnet50'

    htsat_spectrogram_mode = (cli.model_type == 'htsat') and not cli.htsat_waveform_input

    # Dataset setup
    if USE_AUDIO:
        dataset = AudioDataset(args.audio_dir)
        data_loader = create_audio_data_loader(dataset, batch_size=1, num_workers=args.workers)
        preprocessor = get_preprocessor(cli.model_type, config, htsat_spectrogram_mode=htsat_spectrogram_mode)
        if htsat_spectrogram_mode:
            print("Using HTSAT spectrogram preprocessing for saliency masks.")
        else:
            print(f"Using raw audio with {cli.model_type} waveform preprocessing")
        # Update input size from preprocessor output for accurate mask dimensions
        try:
            sample_tensor = preprocessor(dataset.audio_files[0])
            if isinstance(sample_tensor, torch.Tensor):
                args.input_size = tuple(sample_tensor.shape[-2:])
        except Exception as sample_exc:
            print(f"Warning: failed to infer input size from sample ({sample_exc}). Using config defaults.")
    else:
        dataset = CustomImageDataset(args.datadir, preprocess)
        if args.range is not None:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range)
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
        preprocessor = None
        print(f"Using pre-generated spectrograms with {cli.model_type}")

    # Model setup using factory
    model = create_model(
        cli.model_type,
        weights_path=config.get('weights_path'),
        num_classes=config.get('num_classes', 50),
        htsat_spectrogram_mode=htsat_spectrogram_mode,
    )

    # Explainer setup
    explainer = TFStructuredRISE(
        model,
        args.input_size,
        args.gpu_batch,
        soft_masking=cli.soft_masking,
        edge_sigma_px=cli.edge_sigma_px,
        occlusion=cli.occlusion
    )

    # Determine mask fractions based on mask type
    if cli.mask_type == 'time':
        fractions = dict(time=1.0, freq=0.0, rect=0.0, mel=0.0)
    elif cli.mask_type == 'freq':
        fractions = dict(time=0.0, freq=1.0, rect=0.0, mel=0.0)
    elif cli.mask_type == 'rect':
        fractions = dict(time=0.0, freq=0.0, rect=1.0, mel=0.0)
    elif cli.mask_type == 'mel':
        fractions = dict(time=0.0, freq=0.0, rect=0.0, mel=1.0)
    else:  # 'all'
        fractions = dict(
            time=cli.time_stripe_frac,
            freq=cli.freq_band_frac,
            rect=cli.rect_patch_frac,
            mel=cli.mel_band_frac
        )

    # Generate or load masks
    masking_style = f"_{cli.soft_masking}" if cli.soft_masking != "none" else "_discrete"
    mask_suffix = f"_{cli.mask_type}" if cli.mask_type != "all" else "_combined"
    occlusion_suffix = f"_occlusion_{cli.occlusion}"
    
    # Use custom mask name if provided, otherwise use auto-generated
    if cli.mask_name:
        maskspath = f'results/masks/{cli.model_type}/{cli.mask_name}.npy'
    else:
        maskspath = f'results/masks/{cli.model_type}/masks{masking_style}{mask_suffix}.npy'
    
    if cli.generate_new or not os.path.isfile(maskspath):
        # Scale mask parameters based on input dimensions for model-agnostic masking
        H, W = args.input_size
        
        # # Target: masks should cover ~1-5% for time, ~5-15% for frequency
        # # This ensures balanced coverage regardless of spectrogram shape
        # time_stripe_width_px = (max(4, int(W * 0.05)), max(10, int(W * 0.15)))  # 5%-15% of width
        # freq_band_height_px = (max(4, int(H * 0.005)), max(24, int(H * 0.01)))   # .5%-1% of height
        # rect_size_px = (max(8, int(H * 0.04)), max(48, int(W * 0.21)))   # 10% × 3%

        # # Force masks to match HTSAT patch size (approx 4-8 pixels)
        # # Time: ~25-50ms | Freq: ~1-2 mel bins
        # time_stripe_width_px = (max(4, int(W * 0.01)), max(8, int(W * 0.02)))
        # freq_band_height_px = (max(4, int(H * 0.01)), max(8, int(H * 0.02)))
        # # Rectangles: Localized patches (4x4 to 8x8)
        # rect_size_px = (max(4, int(H * 0.01)), max(8, int(W * 0.02)))

        # Masks cover approx 100ms or 5-10 mel bins
        time_stripe_width_px = (max(8, int(W * 0.03)), max(16, int(W * 0.06)))
        freq_band_height_px = (max(4, int(H * 0.05)), max(12, int(H * 0.10)))
        # Rectangles: roughly the size of a distinct sound event
        rect_size_px = (max(8, int(H * 0.05)), max(32, int(W * 0.10)))

        print(f"Generating masks for input size {args.input_size}")
        print(f"  Time stripe width: {time_stripe_width_px} pixels ({100*time_stripe_width_px[0]/W:.1f}%-{100*time_stripe_width_px[1]/W:.1f}% of width)")
        print(f"  Freq band height: {freq_band_height_px} pixels ({100*freq_band_height_px[0]/H:.1f}%-{100*freq_band_height_px[1]/H:.1f}% of height)")
        print(f"  Rect patch size: {rect_size_px} pixels ({100*rect_size_px[0]/H:.1f}% × {100*rect_size_px[1]/W:.1f}%)")
        
        explainer.generate_tf_masks(
            N=cli.N,
            time_stripe_frac=fractions['time'],
            freq_band_frac=fractions['freq'],
            rect_patch_frac=fractions['rect'],
            mel_band_frac=fractions['mel'],
            time_stripe_width_px=time_stripe_width_px,
            freq_band_height_px=freq_band_height_px,
            rect_size_px=rect_size_px,
            rect_count_range=(1, 6),
            stripe_count_range=(1, 12),
            mel_bands=64,
            band_keep_prob=0.3,
            savepath=maskspath,
        )
    else:
        explainer.load_masks(maskspath)
        print(f'Masks loaded from {maskspath}')

    # Generate explanations
    def explain_all(data_loader, explainer, use_audio=False, preprocessor=None):
        target = np.empty(len(data_loader), np.int64)
        filenames = []
        
        for i, (input_data, _, path) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
            if use_audio and preprocessor:
                model_input = preprocessor(input_data[0]).cuda()
            else:
                model_input = input_data.cuda()
            
            pred_full = model(model_input)
            p, c = torch.max(pred_full, dim=1)
            target[i] = c[0]
            filenames.append(path[0])

        explanations = np.empty((len(data_loader), *args.input_size))
        for i, (input_data, _, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
            if use_audio and preprocessor:
                model_input = preprocessor(input_data[0]).cuda()
            else:
                model_input = input_data.cuda()
            
            saliency_maps = explainer(model_input)
            explanations[i] = saliency_maps[target[i]].cpu().numpy()
        
        return explanations, target, filenames

    explanations, targets, filenames = explain_all(data_loader, explainer, USE_AUDIO, preprocessor if USE_AUDIO else None)

    # Save results
    import pickle
    # Use custom output name if provided, otherwise use auto-generated
    # Always organize by model_type first, then by experiment
    if cli.output_name:
        output_dir = f'results/{cli.model_type}/saliency/{cli.output_name}'
    else:
        experiment_name = f'{cli.dataset}_{cli.mask_type}{masking_style}{occlusion_suffix}'
        output_dir = f'results/{cli.model_type}/saliency/{experiment_name}'

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    np.save(os.path.join(output_dir, 'all_saliency_maps.npy'), explanations)

    results = {
        'explanations': explanations,
        'targets': targets,
        'filenames': filenames,
        'dataset_size': len(dataset),
        'input_size': args.input_size,
        'model': cli.model_type,
        'dataset': cli.dataset,
        'use_audio': USE_AUDIO,
        'mask_type': f'tf_structured{mask_suffix}',
        'soft_masking': cli.soft_masking,
        'edge_sigma_px': cli.edge_sigma_px,
        'occlusion': cli.occlusion,
        'mask_fractions': fractions,
        'N': cli.N,
        'mask_file': maskspath,
        'occlusion_method': cli.occlusion
    }

    with open(os.path.join(output_dir, 'saliency_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved saliency maps for {len(dataset)} images to {output_dir}/")

    # Save individual files
    def save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, use_audio=False, preprocessor=None):
        os.makedirs(output_dir, exist_ok=True)
        for i, (input_data, _, path) in enumerate(tqdm(data_loader, desc='Saving individual maps')):
            if use_audio and preprocessor:
                model_input = preprocessor(input_data[0]).cuda()
            else:
                model_input = input_data.cuda()
            
            p, c = torch.max(model(model_input), dim=1)
            p, c = p[0].item(), c[0].item()

            original_path = path[0]
            original_name = os.path.basename(original_path)
            name_without_ext = os.path.splitext(original_name)[0]

            ref_dir = args.audio_dir if use_audio else args.datadir
            rel_path = os.path.relpath(original_path, ref_dir)
            rel_dir = os.path.dirname(rel_path)
            if rel_dir:
                subdir_output = os.path.join(output_dir, rel_dir)
                os.makedirs(subdir_output, exist_ok=True)
                output_path = subdir_output
            else:
                output_path = output_dir

            saliency_map = explanations[i]
            np.save(os.path.join(output_path, f'{name_without_ext}_saliency.npy'), saliency_map)

            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.axis('off')
            plt.title(f'{100 * p:.2f}% {get_class_name(c)}')
            if use_audio:
                tensor_imshow(model_input[0].detach().cpu())
            else:
                tensor_imshow(input_data[0])
            plt.subplot(122)
            plt.axis('off')
            plt.title(get_class_name(c))
            if use_audio:
                tensor_imshow(model_input[0].detach().cpu())
            else:
                tensor_imshow(input_data[0])
            plt.imshow(saliency_map, cmap='jet', alpha=0.5)
            plt.savefig(os.path.join(output_path, f'{name_without_ext}_saliency.png'), bbox_inches='tight', dpi=150)
            plt.close()

    save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, USE_AUDIO, preprocessor if USE_AUDIO else None)


if __name__ == "__main__":
    main()
