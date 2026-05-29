"""
Audio preprocessing for ResNet50 UrbanSound8K

This module is designed to closely match the ESC-50 ResNet50 preprocessing
pipeline used in `ResNet50/utils.py` and `RISE_audio`:

- Load mono audio at 22.05 kHz
- (Optionally) pad/truncate to a fixed duration
- Compute mel spectrogram (128 mel bins)
- Convert to dB scale and min‑max normalise
- Map to [0, 255] and convert to a grayscale PIL image
- Resize to 224×224 so it can be treated as an image by ResNet50

The dataset then applies a transform that:
- Converts the grayscale image to a tensor
- Replicates the single channel to 3 channels
- Applies simple [-1, 1] style normalisation
"""

import numpy as np
import librosa
from PIL import Image


class AudioPreprocessor:
    """
    Audio preprocessor for converting audio files to mel‑spectrogram images
    suitable for ResNet50 input.
    """

    def __init__(
        self,
        sr: int = 22050,
        duration: float = 4.0,
        n_mels: int = 128,
        fmax: float = None,
        hop_length: int = 512,
        n_fft: int = 1024,
        target_width: int = 128,
        image_size=(224, 224),
    ):
        """
        Initialize audio preprocessor.

        Args:
            sr: Target sampling rate (Hz)
            duration: Target audio duration (seconds)
            n_mels: Number of mel bands
            fmax: Maximum frequency (Hz); None = use full Nyquist (sr/2)
            hop_length: Hop length for STFT
            n_fft: FFT window size
            target_width: Kept for backward compatibility (no longer used directly)
            image_size: Output image size (height, width), default (224, 224)
        """
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)
        self.n_mels = n_mels
        self.fmax = fmax
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_width = target_width
        self.image_size = image_size

    def load_audio(self, audio_path):
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            audio: Audio waveform as numpy array
        """
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            return audio
        except Exception as e:
            raise Exception(f"Error loading audio file {audio_path}: {e}")

    def pad_or_truncate(self, audio):
        """
        Pad or truncate audio to target length.

        Args:
            audio: Audio waveform as numpy array

        Returns:
            audio: Audio padded or truncated to target length
        """
        if len(audio) < self.target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode="constant")
        else:
            # Truncate to target length
            audio = audio[: self.target_length]

        return audio

    def compute_mel_spectrogram(self, audio):
        """
        Convert audio waveform to mel spectrogram (normalised to [0, 1]).

        Args:
            audio: Audio waveform as numpy array

        Returns:
            mel_spec_norm: Mel spectrogram in dB scale, min‑max normalised to [0, 1]
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            fmax=self.fmax,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] (same formula as ESC-50 utils)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )

        return mel_spec_norm

    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline: load audio and convert to 224×224
        grayscale mel‑spectrogram image.

        Args:
            audio_path: Path to audio file

        Returns:
            img: PIL.Image.Image, grayscale, resized to `image_size`
        """
        # Load audio
        audio = self.load_audio(audio_path)

        # Pad or truncate to target length
        audio = self.pad_or_truncate(audio)

        # Compute normalised mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)  # (n_mels, time), in [0, 1]

        # Map to [0, 255] and convert to uint8 image
        mel_img_array = (mel_spec * 255.0).astype(np.uint8)

        # Create PIL Image and resize to match ESC-50 / RISE_audio pipeline
        img = Image.fromarray(mel_img_array)
        img = img.convert("L")  # Ensure grayscale
        img = img.resize(self.image_size, Image.BILINEAR)

        return img

    def preprocess_batch(self, audio_paths):
        """
        Preprocess a batch of audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            images: List of PIL grayscale images resized to `image_size`
        """
        images = []
        for audio_path in audio_paths:
            img = self.preprocess(audio_path)
            images.append(img)

        return images
