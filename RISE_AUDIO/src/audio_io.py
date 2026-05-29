import torch
import torchaudio
import os
import glob
import numpy as np
from typing import List, Tuple, Optional, Union
from src.utils import (
    original_nfft, MASKS_DIR, DEFAULT_N_MASKS, DEFAULT_TIME_STRIPE_FRAC,
    DEFAULT_FREQ_BAND_FRAC, DEFAULT_RECT_PATCH_FRAC, DEFAULT_MEL_BAND_FRAC,
    DEFAULT_SOFT_MASKING, DEFAULT_EDGE_SIGMA_PX, DEFAULT_SINGLE_BLOCK, DEFAULT_OCCLUSION, TARGET_ORIGINAL_SR
)
from src.mask_generator import TFMaskGenerator

# TARGET_ORIGINAL_SR = MODEL_SAMPLE_RATES.get(DEFAULT_MODEL_TYPE, 'N/A') * 3

class AudioProcessor:
    """Audio processing class for handling multiple audio files with masking capabilities."""

    def __init__(self, n_fft: int = None, hop_length: int = None, soft_masking: str = None, single_block: bool = None, occlusion: str = None):
        """
        Initialize the audio processor.

        Args:
            n_fft: FFT window size (defaults to original_nfft from utils)
            hop_length: Window shift (defaults to n_fft // 2)
            soft_masking: Type of soft masking to apply (defaults to DEFAULT_SOFT_MASKING)
            single_block: Whether to use single block masking (defaults to DEFAULT_SINGLE_BLOCK)
            occlusion: Type of occlusion to apply (defaults to DEFAULT_OCCLUSION)
        """
        self.n_fft = n_fft or original_nfft
        self.hop_length = hop_length or (self.n_fft // 2)
        self.soft_masking = soft_masking or DEFAULT_SOFT_MASKING
        self.single_block = DEFAULT_SINGLE_BLOCK if single_block is None else single_block
        self.occlusion = occlusion or DEFAULT_OCCLUSION
        self.target_sr = TARGET_ORIGINAL_SR
        # Initialize STFT transforms
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            onesided=True,
            power=None,
        )

        self.inv_stft = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            onesided=True,
        )

        # Initialize mask generator
        self.mask_generator = None
        self.masks = None
        self.mask_file_path = None

        # Create masks directory
        os.makedirs(MASKS_DIR, exist_ok=True)

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to mono.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        waveform, sr = torchaudio.load(file_path, channels_first=True)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
        audio = resampler(waveform)
        # Convert to mono if multi-channel
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, self.target_sr

    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to complex spectrogram.

        Args:
            audio: Audio tensor of shape (1, n_samples)

        Returns:
            Complex spectrogram tensor of shape (1, n_fft//2+1, n_frames)
        """
        return self.stft(audio)

    def complex_to_real_imag(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert complex spectrogram to real and imaginary channels.

        Args:
            spec: Complex spectrogram tensor of shape (1, n_fft//2+1, n_frames)

        Returns:
            Real and imaginary tensor of shape (2, n_fft//2+1, n_frames)
        """
        return torch.cat((spec.real, spec.imag), dim=0)

    def real_imag_to_complex(self, real_imag_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert real and imaginary channels back to complex spectrogram.

        Args:
            real_imag_spec: Real and imaginary tensor of shape (2, n_fft//2+1, n_frames)

        Returns:
            Complex spectrogram tensor of shape (1, n_fft//2+1, n_frames)
        """
        return torch.complex(real_imag_spec[:1], real_imag_spec[1:])

    def spectrogram_to_audio(self, spec: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Convert complex spectrogram back to audio.

        Args:
            spec: Complex spectrogram tensor
            n_samples: Original number of samples

        Returns:
            Audio tensor of shape (1, n_samples)
        """
        return self.inv_stft(spec, n_samples)


    def apply_masking(self, real_imag_spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masking to real and imaginary spectrogram channels with different occlusion types.

        Args:
            real_imag_spec: Real and imaginary tensor of shape (2, n_fft//2+1, n_frames)
            mask: Mask tensor of shape (n_fft//2+1, n_frames) or (1, n_fft//2+1, n_frames)

        Returns:
            Masked real and imaginary tensor
        """
        # Ensure mask has correct shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        elif mask.dim() == 3 and mask.shape[0] != 1:
            mask = mask.unsqueeze(0)  # Add batch dimension if needed

        # Ensure mask matches spectrogram dimensions
        if mask.shape[1:] != real_imag_spec.shape[1:]:
            # Resize mask to match spectrogram dimensions
            print("Resizing mask to match spectrogram dimensions")
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=real_imag_spec.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Apply different occlusion types using vectorized operations
        if self.occlusion == "black":
            # Original behavior: simple multiplication (0 for occluded regions)
            baseline = torch.zeros_like(real_imag_spec)
            
        elif self.occlusion == "freq":
            # Row-wise mean across time (dim=2) for each frequency bin
            # real_imag_spec shape: (2, n_fft//2+1, n_frames)
            # We want to average across time (dim=2) for each frequency bin
            baseline = real_imag_spec.mean(dim=2, keepdim=True).expand_as(real_imag_spec)
            
        elif self.occlusion == "time":
            # Column-wise mean across frequencies (dim=1) for each time step
            # real_imag_spec shape: (2, n_fft//2+1, n_frames)
            # We want to average across frequency (dim=1) for each time step
            baseline = real_imag_spec.mean(dim=1, keepdim=True).expand_as(real_imag_spec)
            
        else:
            # Default to black occlusion for unknown types
            print(f"Unknown occlusion type '{self.occlusion}', defaulting to 'black'")
            baseline = torch.zeros_like(real_imag_spec)

        # Blend input with baseline outside mask
        # mask: (1, n_fft//2+1, n_frames), real_imag_spec/baseline: (2, n_fft//2+1, n_frames)
        # Broadcasting will work: (1, H, W) * (2, H, W) -> (2, H, W)
        masked_real_imag = mask * real_imag_spec + (1.0 - mask) * baseline

        return masked_real_imag

    def generate_or_load_masks(self, spec_shape: Tuple[int, int], n_masks: int = DEFAULT_N_MASKS) -> np.ndarray:
        """
        Generate or load masks for the given spectrogram shape.
        Automatically checks if mask file exists and loads it, otherwise generates new masks.

        Args:
            spec_shape: Shape of spectrogram (height, width)
            n_masks: Number of masks to generate

        Returns:
            Array of masks
        """
        # Create mask filename
        h, w = spec_shape
        mode_tag = "single" if self.single_block else "multiple"
        mask_filename = f"masks_{n_masks}_{h}_{w}_{DEFAULT_SOFT_MASKING}_{mode_tag}_all.npy"
        self.mask_file_path = os.path.join(MASKS_DIR, mask_filename)

        # Check if mask file already exists
        if os.path.exists(self.mask_file_path):
            print(f"Loading existing masks from: {self.mask_file_path}")
            self.masks = np.load(self.mask_file_path)
            return self.masks

        # Generate new masks if file doesn't exist
        print(f"Generating {n_masks} masks with all options...")
        print(f"Parameters:")
        print(f"  - Time stripe fraction: {DEFAULT_TIME_STRIPE_FRAC}")
        print(f"  - Frequency band fraction: {DEFAULT_FREQ_BAND_FRAC}")
        print(f"  - Rectangular patch fraction: {DEFAULT_RECT_PATCH_FRAC}")
        print(f"  - Mel band fraction: {DEFAULT_MEL_BAND_FRAC}")
        print(f"  - Soft masking: {DEFAULT_SOFT_MASKING}")
        print(f"  - Single Block: {DEFAULT_SINGLE_BLOCK}")
        print(f"  - Edge sigma: {DEFAULT_EDGE_SIGMA_PX}")
        print(f"  - Occlusion type: {self.occlusion}")

        if self.mask_generator is None:
            self._initialize_mask_generator(spec_shape)

        # Generate masks with all parameters from rise_utils.py
        self.masks = self.mask_generator.generate_tf_masks(
            N=n_masks,
            time_stripe_frac=DEFAULT_TIME_STRIPE_FRAC,
            freq_band_frac=DEFAULT_FREQ_BAND_FRAC,
            rect_patch_frac=DEFAULT_RECT_PATCH_FRAC,
            mel_band_frac=DEFAULT_MEL_BAND_FRAC,
            single_block=DEFAULT_SINGLE_BLOCK,
            savepath=self.mask_file_path
        )

        print(f"Masks saved to: {self.mask_file_path}")
        return self.masks

    # def generate_all_masked_audio(self, input_path: str, n_masks: int = DEFAULT_N_MASKS) -> Tuple[
    #     List[torch.Tensor], int]:
    #     """
    #     Generate all masked versions of a single audio file using all 6000 masks.

    #     Args:
    #         input_path: Path to input audio file
    #         n_masks: Number of masks to apply (defaults to DEFAULT_N_MASKS = 6000)

    #     Returns:
    #         Tuple of (list_of_masked_audio_tensors, sample_rate)
    #     """
    #     print(f"Generating {n_masks} masked versions of: {os.path.basename(input_path)}")
    #     print(f"Using occlusion type: {self.occlusion}")

    #     # Load audio
    #     audio, sr = self.load_audio(input_path)
    #     n_samples = audio.shape[1]

    #     # Convert to spectrogram
    #     spec = self.audio_to_spectrogram(audio)

    #     # Convert to real and imaginary channels
    #     real_imag_spec = self.complex_to_real_imag(spec)

    #     # Generate or load masks
    #     if self.masks is None:
    #         self.generate_or_load_masks(spec.shape[1:], n_masks)

    #     # Generate all masked audio versions
    #     masked_audio_list = []

    #     for i in range(n_masks):
    #         if i % 1000 == 0:
    #             print(f"Processing mask {i + 1}/{n_masks}")

    #         # Get mask
    #         mask = torch.from_numpy(self.masks[i, 0])  # Remove batch and channel dims

    #         # Apply mask to real and imaginary channels
    #         masked_real_imag = self.apply_masking(real_imag_spec, mask)

    #         # Convert back to complex spectrogram
    #         masked_spec = self.real_imag_to_complex(masked_real_imag)

    #         # Convert back to audio
    #         masked_audio = self.spectrogram_to_audio(masked_spec, n_samples)

    #         masked_audio_list.append(masked_audio)

    #     print(f"Generated {len(masked_audio_list)} masked audio versions")
    #     return masked_audio_list, sr, spec.shape[1:]

    def generate_all_masked_audio_vectorized(self, input_path: str, n_masks: int = DEFAULT_N_MASKS, batch_size: int = 100) -> Tuple[List[torch.Tensor], int]:
        """Vectorized version - process masks in batches."""
        # Load audio once
        audio, sr = self.load_audio(input_path)
        n_samples = audio.shape[1]
        
        # Convert to spectrogram once
        spec = self.audio_to_spectrogram(audio)
        real_imag_spec = self.complex_to_real_imag(spec)
        
        # Generate or load masks
        if self.masks is None:
            self.generate_or_load_masks(spec.shape[1:], n_masks)
        
        masked_audio_list = []
        
        # Process masks in batches
        for batch_start in range(0, n_masks, batch_size):
            batch_end = min(batch_start + batch_size, n_masks)
            batch_masks = self.masks[batch_start:batch_end]  # (batch_size, 1, H, W)
            
            # Expand real_imag_spec to match batch: (2, H, W) -> (batch_size, 2, H, W)
            batch_real_imag = real_imag_spec.unsqueeze(0).expand(batch_end - batch_start, -1, -1, -1)
            
            # Apply masks in batch: (batch_size, 1, H, W) * (batch_size, 2, H, W)
            batch_masks_tensor = torch.from_numpy(batch_masks).float()
            if torch.cuda.is_available():
                batch_masks_tensor = batch_masks_tensor.cuda()
                batch_real_imag = batch_real_imag.cuda()
            
            # Apply masking to entire batch
            batch_masked_real_imag = []
            for i in range(batch_end - batch_start):
                masked = self.apply_masking(batch_real_imag[i], batch_masks_tensor[i, 0])
                batch_masked_real_imag.append(masked)
            
            # Convert back to audio for each
            for masked_ri in batch_masked_real_imag:
                masked_spec = self.real_imag_to_complex(masked_ri)
                if masked_spec.is_cuda:
                    masked_spec = masked_spec.cpu()
                masked_audio = self.spectrogram_to_audio(masked_spec, n_samples)
                masked_audio_list.append(masked_audio)
        
        return masked_audio_list, sr, spec.shape[1:]

    def get_mask(self, index: int) -> torch.Tensor:
        """
        Get a specific mask by index.

        Args:
            index: Index of the mask to retrieve

        Returns:
            Mask tensor
        """
        if self.masks is None:
            raise ValueError("No masks loaded. Call generate_or_load_masks() first.")

        # Cycle through masks if index exceeds available masks
        actual_index = index % len(self.masks)
        return torch.from_numpy(self.masks[actual_index, 0])  # Remove batch and channel dims

    def process_single_file(self, input_path: str, output_path: str = None,
                            mask_index: int = None, apply_masking: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Process a single audio file with optional masking.

        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio (optional)
            mask_index: Index of mask to use (optional, will use random if not provided)
            apply_masking: Whether to apply masking

        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Load audio
        audio, sr = self.load_audio(input_path)
        n_samples = audio.shape[1]

        # Convert to spectrogram
        spec = self.audio_to_spectrogram(audio)

        # Convert to real and imaginary channels
        real_imag_spec = self.complex_to_real_imag(spec)

        # Apply masking if requested
        if apply_masking:
            # Generate or load masks if not already done
            if self.masks is None:
                self.generate_or_load_masks(spec.shape[1:])

            # Get mask
            if mask_index is None:
                mask_index = np.random.randint(0, len(self.masks))

            mask = self.get_mask(mask_index)

            # Apply mask to real and imaginary channels
            real_imag_spec = self.apply_masking(real_imag_spec, mask)

        # Convert back to complex spectrogram
        new_spec = self.real_imag_to_complex(real_imag_spec)

        # Convert back to audio
        processed_audio = self.spectrogram_to_audio(new_spec, n_samples)

        # Save if output path provided
        if output_path:
            torchaudio.save(output_path, processed_audio, sample_rate=sr)

        return processed_audio, sr

    def process_directory(self, input_dir: str, output_dir: str = None,
                          file_pattern: str = "*.wav", apply_masking: bool = True,
                          mask_per_file: bool = True) -> List[Tuple[str, torch.Tensor, int]]:
        """
        Process multiple audio files from a directory.

        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed files (optional)
            file_pattern: File pattern to match (e.g., "*.wav", "*.mp3")
            apply_masking: Whether to apply masking
            mask_per_file: Whether to use a different mask for each file

        Returns:
            List of tuples (filename, processed_audio, sample_rate)
        """
        # Find all matching files
        search_pattern = os.path.join(input_dir, file_pattern)
        file_paths = glob.glob(search_pattern)

        if not file_paths:
            print(f"No files found matching pattern: {search_pattern}")
            return []

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []

        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")

            # Generate output path if output directory specified
            output_path = None
            if output_dir:
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_processed{ext}")

            try:
                # Process file with mask index if mask_per_file is True
                mask_index = i if mask_per_file else None
                processed_audio, sr = self.process_single_file(
                    file_path,
                    output_path,
                    mask_index=mask_index,
                    apply_masking=apply_masking
                )

                results.append((filename, processed_audio, sr))
                print(f"Successfully processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        return results

    def _initialize_mask_generator(self, spec_shape: Tuple[int, int]):
        """Initialize mask generator with spectrogram dimensions."""
        self.mask_generator = TFMaskGenerator(
            input_size=spec_shape,
            soft_masking=DEFAULT_SOFT_MASKING,
            edge_sigma_px=DEFAULT_EDGE_SIGMA_PX
        )


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = AudioProcessor()

    # Generate all 6000 masked versions of a single file
    input_file = "__8ompVQG6M_30.wav"
    if os.path.exists(input_file):
        # masked_audio_list, sr = processor.generate_all_masked_audio(input_file)
        masked_audio_list, sr = processor.generate_all_masked_audio_vectorized(input_file)
        print(f"Generated {len(masked_audio_list)} masked versions of {input_file}")
        print(f"Each audio tensor shape: {masked_audio_list[0].shape}")