#!/usr/bin/env python
# coding: utf-8

"""Saliency map generation for RISE audio XAI - adapted from RISE_Spec_Waveform_framework."""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms

from src.mask_generator import TFMaskGenerator
from src.feature_extractor import process_masked_audio_list


def pad_or_truncate_waveform(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pad or truncate waveform to target length.
    
    Args:
        waveform: Input waveform tensor (1, samples) or (samples,)
        target_length: Target number of samples
        
    Returns:
        Waveform padded or truncated to target_length
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    current_length = waveform.shape[1]
    
    if current_length > target_length:
        # Center crop
        start = (current_length - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif current_length < target_length:
        # Pad with zeros
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    
    return waveform


class RISEAudioSaliency:
    """RISE-style saliency generator for audio using mel spectrogram perturbations."""

    def __init__(
            self,
            model,
            input_size: Tuple[int, int] = (224, 224),
            gpu_batch: int = 250,
            soft_masking: str = "gaussian",
            edge_sigma_px: float = 1.0,
            occlusion: str = "black"
    ):
        """Initialize RISE audio saliency generator.

        Args:
            model: Model instance with predict method
            input_size: Size of mel spectrogram images (H, W)
            gpu_batch: Batch size for GPU processing
            soft_masking: Soft masking method
            edge_sigma_px: Gaussian blur sigma
            occlusion: Occlusion baseline method
        """
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.occlusion = occlusion
        self.masks = None
        self.p1 = None
        self.N = None

        # Initialize mask generator
        self.mask_generator = TFMaskGenerator(
            input_size=input_size,
            soft_masking=soft_masking,
            edge_sigma_px=edge_sigma_px
        )

        # Image preprocessing transform
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ImageNet normalization
        ])

    def load_masks(self, mask_path: str):
        """Load pre-generated masks from file."""
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Masks file not found: {mask_path}")

        masks = np.load(mask_path)
        # Convert from [N, 1, H, W] to [N, 3, H, W] for RGB images
        # if masks.shape[1] == 1:
        #     masks = np.repeat(masks, 3, axis=1)  # Repeat across 3 channels

        self.masks = torch.from_numpy(masks).float()

        if torch.cuda.is_available():
            self.masks = self.masks.cuda()

        self.N = self.masks.shape[0]
        self.p1 = float(self.masks.mean().item())
        print(f"Loaded {self.N} masks from {mask_path}")

    # def generate_masks(
    #         self,
    #         N: int,
    #         time_stripe_frac: float = 0.25,
    #         freq_band_frac: float = 0.25,
    #         rect_patch_frac: float = 0.25,
    #         mel_band_frac: float = 0.25,
    #         savepath: Optional[str] = None
    # ) -> np.ndarray:
    #     """Generate masks using TFMaskGenerator."""
    #     masks = self.mask_generator.generate_tf_masks(
    #         N=N,
    #         time_stripe_frac=time_stripe_frac,
    #         freq_band_frac=freq_band_frac,
    #         rect_patch_frac=rect_patch_frac,
    #         mel_band_frac=mel_band_frac,
    #         savepath=savepath
    #     )
    #
    #     # Convert to torch tensor
    #     self.masks = torch.from_numpy(masks).float()
    #     if torch.cuda.is_available():
    #         self.masks = self.masks.cuda()
    #
    #     self.N = self.masks.shape[0]
    #     self.p1 = float(self.masks.mean().item())
    #
    #     return masks

    def generate_saliency_from_preprocessed_inputs(
            self,
            preprocessed_inputs: List,
            target_class: Optional[int] = None,
            progress_callback: Optional[callable] = None,
            orig_input = None
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Generate saliency map from preprocessed inputs (model-specific format).
        
        This is the unified method that handles inputs preprocessed by model-specific
        preprocessors (PIL Images for ResNet, tensors for wav2vec2, etc.)

        Args:
            preprocessed_inputs: List of preprocessed inputs (format depends on model)
            target_class: Target class to explain (if None, uses predicted class)
            progress_callback: Optional progress callback
            orig_input: Unmasked original input to determine target class (required if target_class is None)
        """
        if self.masks is None:
            raise ValueError("No masks loaded. Call load_masks() or generate_masks() first.")

        N = len(preprocessed_inputs)

        # Get target class from original input
        if target_class is None:
            if orig_input is None:
                raise ValueError("orig_input must be provided to determine target class.")
            
            # Convert to model input format
            orig_tensor = self._to_model_input(orig_input)
            if torch.cuda.is_available():
                orig_tensor = orig_tensor.cuda()

            with torch.no_grad():
                logits, probs = self.model.predict(orig_tensor)
                target_class = torch.argmax(probs).item()

        # Process inputs in batches
        chunk_size = min(self.gpu_batch, N)
        predictions = []

        for i in tqdm(range(0, N, chunk_size), desc="Processing inputs"):
            end_idx = min(i + chunk_size, N)
            chunk_inputs = preprocessed_inputs[i:end_idx]

            # Batch inputs to tensor
            chunk_tensor = self._batch_inputs(chunk_inputs)
            if torch.cuda.is_available():
                chunk_tensor = chunk_tensor.cuda()

            # Get model predictions
            with torch.no_grad():
                logits, probs = self.model.predict(chunk_tensor)
                predictions.append(probs)

            if progress_callback:
                progress_callback("Processing inputs", end_idx, N)

        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=0)  # Shape: (N, num_classes)

        # Extract target class scores
        target_scores = predictions[:, target_class].detach().cpu().numpy()

        # Compute saliency using RISE formula
        mask_H, mask_W = self.masks.shape[2], self.masks.shape[3]
        saliency = self._compute_saliency(target_scores, mask_H, mask_W)

        # Metadata
        metadata = {
            "num_masks": N,
            "target_class": target_class,
            "input_size": self.input_size,
            "mask_probability": self.p1,
            "mean_score": float(np.mean(target_scores)),
            "std_score": float(np.std(target_scores)),
            "occlusion_type": self.occlusion
        }

        return saliency, target_class, metadata
    
    def _to_model_input(self, preprocessed_input):
        """Convert single preprocessed input to model input tensor."""
        if isinstance(preprocessed_input, Image.Image):
            # ResNet case: PIL Image → Tensor with transforms
            return self.transform(preprocessed_input).unsqueeze(0)
        elif isinstance(preprocessed_input, torch.Tensor):
            # Raw audio models: already a tensor
            if preprocessed_input.dim() == 1:
                return preprocessed_input.unsqueeze(0)  # (samples,) → (1, samples)
            elif preprocessed_input.dim() == 2 and preprocessed_input.shape[0] != 1:
                return preprocessed_input.unsqueeze(0)  # (channels, samples) → (1, channels, samples)
            return preprocessed_input
        else:
            raise ValueError(f"Unknown input type: {type(preprocessed_input)}")
    
    def _batch_inputs(self, inputs):
        """Batch multiple inputs into single tensor."""
        if isinstance(inputs[0], Image.Image):
            # ResNet: PIL Images → RGB tensors
            tensors = [self.transform(img) for img in inputs]
            return torch.stack(tensors)
        elif isinstance(inputs[0], torch.Tensor):
            # Raw audio models: stack tensors
            # Handle different tensor shapes
            if inputs[0].dim() == 1:
                # (samples,) → stack to (batch, samples)
                return torch.stack(inputs)
            elif inputs[0].dim() == 2:
                # Check if first dimension is 1 (channel dimension that should be removed)
                if inputs[0].shape[0] == 1:
                    # (1, samples) → squeeze to (samples,) → stack to (batch, samples)
                    # This is needed for HTSAT and similar models
                    squeezed = [inp.squeeze(0) for inp in inputs]
                    return torch.stack(squeezed)
                else:
                    # (channels, samples) → stack to (batch, channels, samples)
                    return torch.stack(inputs)
            else:
                return torch.stack(inputs)
        else:
            raise ValueError(f"Unknown input type: {type(inputs[0])}")
    
    def generate_saliency_from_mel_images(
            self,
            mel_images: List[Image.Image],
            target_class: Optional[int] = None,
            progress_callback: Optional[callable] = None,
            orig_mel_img: Image.Image = None
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        DEPRECATED: Use generate_saliency_from_preprocessed_inputs instead.
        
        Generate saliency map from mel spectrogram images using RISE methodology.
        This method is kept for backward compatibility.
        """
        return self.generate_saliency_from_preprocessed_inputs(
            preprocessed_inputs=mel_images,
            target_class=target_class,
            progress_callback=progress_callback,
            orig_input=orig_mel_img
        )

    def _compute_saliency(
            self,
            scores: np.ndarray,
            H: int = None,  # Make optional
            W: int = None   # Make optional
    ) -> np.ndarray:
        """Compute saliency map using RISE formula.

        Args:
            scores: Prediction scores of shape (N,)
            H: Spectrogram height (optional, will use mask dimensions if not provided)
            W: Spectrogram width (optional, will use mask dimensions if not provided)

        Returns:
            Saliency map of shape (H, W)
        """
        # Use actual mask dimensions if H, W not provided
        if H is None or W is None:
            mask_H, mask_W = self.masks.shape[2], self.masks.shape[3]
        else:
            mask_H, mask_W = H, W
        
        # Reshape masks to (N, mask_H * mask_W)
        masks_flat = self.masks.view(self.N, mask_H * mask_W)

        # Convert scores to tensor
        scores_tensor = torch.from_numpy(scores).float()
        if torch.cuda.is_available():
            scores_tensor = scores_tensor.cuda()

        # Compute saliency using RISE formula: S = (M^T * P) / (N * p1)
        saliency = torch.matmul(scores_tensor, masks_flat)
        saliency = saliency.view(mask_H, mask_W)
        saliency = saliency / self.N / self.p1

        return saliency.cpu().numpy()

    def save_saliency(
            self,
            saliency_map: np.ndarray,
            metadata: Dict[str, Any],
            output_path: str
    ):
        """Save saliency map and metadata."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save saliency map
        np.save(output_path.replace('.npz', '_saliency.npy'), saliency_map)

        # Save metadata
        np.savez(
            output_path,
            saliency_map=saliency_map,
            metadata=metadata
        )

        print(f"Saved saliency map to {output_path}")


def explain_audio_with_mel_images(
        audio_path: str,
        model,
        mask_dir: str,
        output_dir: str,
        n_masks: int = 6000,
        gpu_batch: int = 250,
        soft_masking: str = "gaussian",
        edge_sigma_px: float = 1.0,
        occlusion: str = "black",
        time_stripe_frac: float = 0.25,
        freq_band_frac: float = 0.25,
        rect_patch_frac: float = 0.25,
        mel_band_frac: float = 0.25,
        target_mel_sr: int = 22050,
        single_block:bool = False

) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Explain audio using spectrogram-based model (ResNet).
    
    Uses model-specific preprocessing via model.preprocessor.

    Args:
        audio_path: Path to audio file
        model: Model instance with preprocessor attribute
        mask_dir: Directory containing masks (unused, kept for compatibility)
        output_dir: Output directory
        n_masks: Number of masks
        gpu_batch: GPU batch size
        soft_masking: Soft masking method
        edge_sigma_px: Gaussian blur sigma
        occlusion: Occlusion baseline method
        time_stripe_frac: Fraction of time stripe masks
        freq_band_frac: Fraction of frequency band masks
        rect_patch_frac: Fraction of rectangular patch masks
        mel_band_frac: Fraction of mel band masks
        target_mel_sr: Target mel sample rate (unused, uses model's preprocessor)
        single_block: Single block masking

    Returns:
        Tuple of (saliency_map, target_class, metadata)
    """
    from src.audio_io import AudioProcessor

    # Get model's preprocessor
    if not hasattr(model, 'preprocessor'):
        raise ValueError("Model must have 'preprocessor' attribute")
    
    preprocessor = model.preprocessor
    
    print(f"Using {type(preprocessor).__name__} for preprocessing")
    print(f"Target sample rate: {preprocessor.target_sample_rate} Hz")

    # Initialize audio processor (handles masking in spectral domain)
    audio_processor = AudioProcessor(occlusion=occlusion)
    
    # Load original audio
    orig_audio, orig_sr = audio_processor.load_audio(audio_path)

    # Generate masked audio waveforms (masking in spectral domain)
    masked_audio_list, sample_rate, mask_shape = \
        audio_processor.generate_all_masked_audio_vectorized(audio_path, n_masks)
    
    # Convert to model-specific format using model's preprocessor
    print(f"Converting {len(masked_audio_list)} masked audio to model format...")
    preprocessed_inputs = preprocessor.process_masked_audio_list(
        masked_audio_list, 
        sample_rate
    )
    orig_input = preprocessor.process_original_audio(orig_audio, orig_sr)
    
    # Free masked_audio_list memory immediately after preprocessing
    del masked_audio_list
    del orig_audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("✓ Freed masked audio memory")

    # Initialize saliency generator
    saliency_gen = RISEAudioSaliency(
        model=model,
        input_size=(224, 224),
        gpu_batch=gpu_batch,
        soft_masking=soft_masking,
        edge_sigma_px=edge_sigma_px,
        occlusion=occlusion
    )

    # Load masks
    mask_path = audio_processor.mask_file_path
    saliency_gen.load_masks(mask_path=mask_path)

    # Generate saliency using new unified method
    saliency_map, target_class, metadata = \
        saliency_gen.generate_saliency_from_preprocessed_inputs(
            preprocessed_inputs=preprocessed_inputs,
            orig_input=orig_input
        )
    
    # Free preprocessed inputs after saliency computation
    del preprocessed_inputs
    del orig_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("✓ Freed preprocessed inputs memory")

    # Save results
    output_path = os.path.join(output_dir, f"{os.path.basename(audio_path)}.npz")
    saliency_gen.save_saliency(saliency_map, metadata, output_path)

    return saliency_map, target_class, metadata


def explain_audio_with_raw_waveforms(
        audio_path: str,
        model,
        mask_dir: str,
        output_dir: str,
        n_masks: int = 6000,
        gpu_batch: int = 250,
        soft_masking: str = "gaussian",
        edge_sigma_px: float = 1.0,
        occlusion: str = "black",
        time_stripe_frac: float = 0.25,
        freq_band_frac: float = 0.25,
        rect_patch_frac: float = 0.25,
        mel_band_frac: float = 0.25,
        single_block: bool = False
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Explain audio using raw waveform models (wav2vec2, ACDNet).
    
    Uses model-specific preprocessing via model.preprocessor.
    
    Pipeline:
    1. Generate masked audio in spectral domain (STFT → mask → ISTFT)
    2. Use model's preprocessor to convert to expected format
    3. Feed waveforms to model and compute RISE saliency
    
    Args:
        audio_path: Path to audio file
        model: Raw audio model with preprocessor attribute
        mask_dir: Directory containing masks (unused, kept for compatibility)
        output_dir: Output directory for saving results
        n_masks: Number of masks
        gpu_batch: Batch size for GPU processing
        soft_masking: Soft masking method
        edge_sigma_px: Gaussian blur sigma
        occlusion: Occlusion baseline method
        time_stripe_frac: Fraction of time stripe masks
        freq_band_frac: Fraction of frequency band masks
        rect_patch_frac: Fraction of rectangular patch masks
        mel_band_frac: Fraction of mel band masks
        single_block: Whether to use single block masking
    
    Returns:
        Tuple of (saliency_map, target_class, metadata)
    """
    from src.audio_io import AudioProcessor
    
    print("=" * 70)
    print("Using RAW WAVEFORM pipeline")
    print("=" * 70)
    
    # Get model's preprocessor
    if not hasattr(model, 'preprocessor'):
        raise ValueError("Model must have 'preprocessor' attribute")
    
    preprocessor = model.preprocessor
    model_sr = preprocessor.target_sample_rate
    model_input_length = getattr(preprocessor, 'fixed_length', None)
    
    print(f"Using {type(preprocessor).__name__} for preprocessing")
    print(f"Target sample rate: {model_sr} Hz")
    if model_input_length is not None:
        print(f"Fixed input length: {model_input_length} samples")
    
    # Initialize audio processor (handles masking in spectral domain)
    audio_processor = AudioProcessor(occlusion=occlusion)
    
    # Step 1: Load original audio
    print("\nStep 1: Loading original audio...")
    orig_audio, orig_sr = audio_processor.load_audio(audio_path)
    print(f"Loaded audio: {orig_audio.shape}, SR: {orig_sr} Hz")
    
    # Step 2: Generate masked audio versions
    print(f"\nStep 2: Generating {n_masks} masked audio versions...")
    masked_audio_list, sample_rate, mask_shape = \
        audio_processor.generate_all_masked_audio_vectorized(audio_path, n_masks)
    print(f"Generated {len(masked_audio_list)} masked waveforms")
    print(f"Mask shape (spectral domain): {mask_shape}")
    
    # Step 3: Use model's preprocessor to convert audio
    print(f"\nStep 3: Converting audio using model's preprocessor...")
    preprocessed_inputs = preprocessor.process_masked_audio_list(
        masked_audio_list, 
        sample_rate
    )
    orig_input = preprocessor.process_original_audio(orig_audio, orig_sr)
    print(f"Converted to model format: {type(preprocessed_inputs[0])}")
    
    # Free masked_audio_list memory immediately after preprocessing
    del masked_audio_list
    del orig_audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("✓ Freed masked audio memory")
    
    # Step 4: Initialize saliency generator
    print("\nStep 4: Initializing RISE saliency generator...")
    saliency_gen = RISEAudioSaliency(
        model=model,
        input_size=(224, 224),
        gpu_batch=gpu_batch,
        soft_masking=soft_masking,
        edge_sigma_px=edge_sigma_px,
        occlusion=occlusion
    )
    
    # Load masks
    mask_path = audio_processor.mask_file_path
    saliency_gen.load_masks(mask_path=mask_path)
    
    # Step 5: Generate saliency using unified method
    print("\nStep 5: Computing RISE saliency map...")
    saliency_map, target_class, metadata = \
        saliency_gen.generate_saliency_from_preprocessed_inputs(
            preprocessed_inputs=preprocessed_inputs,
            orig_input=orig_input
        )
    
    # Free preprocessed inputs after saliency computation
    del preprocessed_inputs
    del orig_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✓ Freed preprocessed inputs memory")
    
    # Add raw audio specific metadata
    metadata["model_type"] = "raw_audio"
    metadata["model_sample_rate"] = model_sr
    metadata["masking_sample_rate"] = sample_rate
    
    # Step 6: Save results
    print(f"\nStep 6: Saving results...")
    output_path = os.path.join(output_dir, f"{os.path.basename(audio_path)}.npz")
    saliency_gen.save_saliency(saliency_map, metadata, output_path)
    
    print("=" * 70)
    print("Raw waveform pipeline completed successfully!")
    print("=" * 70)
    
    return saliency_map, target_class, metadata


def explain_audio_unified(
        audio_path: str,
        model,
        mask_dir: str,
        output_dir: str,
        n_masks: int = 6000,
        gpu_batch: int = 250,
        soft_masking: str = "gaussian",
        edge_sigma_px: float = 1.0,
        occlusion: str = "black",
        time_stripe_frac: float = 0.25,
        freq_band_frac: float = 0.25,
        rect_patch_frac: float = 0.25,
        mel_band_frac: float = 0.25,
        target_mel_sr: int = 22050,
        single_block: bool = False
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    UNIFIED ENTRY POINT for audio explanation.
    
    Automatically detects model type and routes to appropriate pipeline:
    - SPECTROGRAM models (ResNet, AST): Uses mel spectrogram images
    - RAW_AUDIO models (wav2vec2): Uses raw waveforms
    
    Args:
        audio_path: Path to audio file
        model: Any model instance (ResNet, wav2vec2, AST, etc.)
        mask_dir: Directory containing masks
        output_dir: Output directory
        n_masks: Number of masks
        gpu_batch: GPU batch size
        soft_masking: Soft masking method
        edge_sigma_px: Gaussian blur sigma
        occlusion: Occlusion baseline method
        time_stripe_frac: Fraction of time stripe masks
        freq_band_frac: Fraction of frequency band masks
        rect_patch_frac: Fraction of rectangular patch masks
        mel_band_frac: Fraction of mel band masks
        target_mel_sr: Target sample rate for mel spectrograms
        single_block: Whether to use single block masking
    
    Returns:
        Tuple of (saliency_map, target_class, metadata)
    """
    from src.models.base import ModelInputType, get_model_type
    
    # Auto-detect model type
    model_type = get_model_type(model)
    
    print("\n" + "=" * 70)
    print(f"UNIFIED SALIENCY PIPELINE")
    print(f"Detected model type: {model_type.value.upper()}")
    print("=" * 70)
    
    if model_type == ModelInputType.SPECTROGRAM:
        # Route to mel spectrogram pipeline
        print("→ Routing to MEL SPECTROGRAM pipeline\n")
        return explain_audio_with_mel_images(
            audio_path=audio_path,
            model=model,
            mask_dir=mask_dir,
            output_dir=output_dir,
            n_masks=n_masks,
            gpu_batch=gpu_batch,
            soft_masking=soft_masking,
            edge_sigma_px=edge_sigma_px,
            occlusion=occlusion,
            time_stripe_frac=time_stripe_frac,
            freq_band_frac=freq_band_frac,
            rect_patch_frac=rect_patch_frac,
            mel_band_frac=mel_band_frac,
            target_mel_sr=target_mel_sr,
            single_block=single_block
        )
    else:
        # Route to raw waveform pipeline
        print("→ Routing to RAW WAVEFORM pipeline\n")
        return explain_audio_with_raw_waveforms(
            audio_path=audio_path,
            model=model,
            mask_dir=mask_dir,
            output_dir=output_dir,
            n_masks=n_masks,
            gpu_batch=gpu_batch,
            soft_masking=soft_masking,
            edge_sigma_px=edge_sigma_px,
            occlusion=occlusion,
            time_stripe_frac=time_stripe_frac,
            freq_band_frac=freq_band_frac,
            rect_patch_frac=rect_patch_frac,
            mel_band_frac=mel_band_frac,
            single_block=single_block
        )