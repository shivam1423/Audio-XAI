import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import librosa
from typing import Optional

# -----------------------------------------------------------------------------
# This file REPLICATES `RISE_original/utils.py` **with minimum changes** so that
# the rest of the RISE pipeline can be reused for AUDIO spectrogram inputs.
# -----------------------------------------------------------------------------

__all__ = [
    "Dummy",
    "audio_to_mel_spectrogram_image",
    "read_tensor",
    "tensor_imshow",
    "ESC50SpectrogramDataset",
    "get_class_name",
    "preprocess",
    "RangeSampler",
]


# Dummy class to store arguments ------------------------------------------------------------------
class Dummy:
    """A simple container exactly like in the original RISE code."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ----------------------------------------------------------------------------
#   Audio ⇾ Mel Spectrogram ⇾ PIL Image helper
# ----------------------------------------------------------------------------

def audio_to_mel_spectrogram_image(
    filepath: str,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> Image.Image:
    """Converts an audio file to a mel-spectrogram PIL image (grayscale).
    
    This function replicates the exact preprocessing pipeline used in
    RISE_audio/src/preprocessor.py (ResNetPreprocessor) to ensure consistency
    between training and evaluation.

    Parameters
    ----------
    filepath : str
        Path to the .wav file.
    sr : int, optional
        Sample rate to load the audio at, by default 22050.
    n_fft : int, optional
        FFT window size for STFT.
    hop_length : int, optional
        Hop length between successive frames.
    n_mels : int, optional
        Number of mel bands.

    Returns
    -------
    PIL.Image.Image
        Grayscale spectrogram resized to 224×224 so that it can be treated as
        an image by ResNet50.
    
    Notes
    -----
    Pipeline matches RISE_audio ResNetPreprocessor:
    1. Load audio at target sample rate (22050 Hz)
    2. Generate mel spectrogram (128 mel bins)
    3. Convert to dB scale
    4. Normalize to [0, 255]
    5. Create PIL Image and resize to 224x224 with BILINEAR interpolation
    """
    # 1. Load audio
    y, sr = librosa.load(filepath, sr=sr)

    # 2. Mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    # 3. Convert to dB scale
    S_db = librosa.power_to_db(S, ref=np.max)

    # 4. Normalize to [0, 255] for image conversion
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_img = (S_norm * 255).astype(np.uint8)

    # 5. Create PIL Image and resize with explicit BILINEAR interpolation
    img = Image.fromarray(S_img)
    img = img.convert("L")  # Ensure grayscale
    img = img.resize((224, 224), Image.BILINEAR)
    
    return img


# ----------------------------------------------------------------------------
#   read_tensor – replicates the signature of the original helper
# ----------------------------------------------------------------------------

read_tensor = transforms.Compose(
    [
        audio_to_mel_spectrogram_image,
        transforms.ToTensor(),  # gives 1×224×224
        # replicate to 3 channels so that pretrained ResNet50 can be reused as-is
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # very simple normalization (0-1 ⇒ -1 to 1). Feel free to adapt.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),  # add batch dim
    ]
)


# ----------------------------------------------------------------------------
#   Visualisation helpers
# ----------------------------------------------------------------------------

def tensor_imshow(inp: torch.Tensor, title: Optional[str] = None, **kwargs):
    """Imshow for Tensor (expects tensor **before** normalization)."""
    inp = inp.numpy().transpose((1, 2, 0))  # C×H×W → H×W×C
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # denormalise
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp.squeeze(), cmap="viridis", **kwargs)
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------------
#   ESC-50 Dataset wrapper
# ----------------------------------------------------------------------------

class ESC50SpectrogramDataset(Dataset):
    """A tiny wrapper that generates log-mel spectrograms on the fly.

    This mirrors torchvision.datasets.ImageFolder used in the original RISE
    script, exposing `.classes` and returning (tensor, label) tuples so that
    downstream code remains unchanged.
    """

    def __init__(self, esc50_root: str, csv_name: str = "meta/esc50.csv", transform=None):
        super().__init__()
        self.audio_dir = os.path.join(esc50_root, "audio")
        self.df = pd.read_csv(os.path.join(esc50_root, csv_name))
        self.transform = transform or read_tensor
        # Build mapping from class id (target) to category string
        self.targets = self.df["target"].to_numpy()
        self.fnames = self.df["filename"].to_numpy()
        self.classes = [
            c for _, c in sorted(
                {(row.target, row.category) for row in self.df.itertuples()}
            )
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.audio_dir, self.fnames[idx])
        label = int(self.targets[idx])
        tensor = self.transform(wav_path)  # 1×3×224×224
        return tensor.squeeze(0), label


# ----------------------------------------------------------------------------
#   Class-name helper
# ----------------------------------------------------------------------------

def get_class_name(c: int) -> str:
    """Returns ESC-50 category string given integer target."""
    # lazily load mapping the first time the function is called
    if not hasattr(get_class_name, "_mapping"):
        # Build mapping one-off
        esc50_csv = os.path.join(os.path.dirname(__file__), "ESC50", "meta", "esc50.csv")
        df = pd.read_csv(esc50_csv)
        mapping = df.drop_duplicates("target").set_index("target")["category"].to_dict()
        get_class_name._mapping = mapping
    return get_class_name._mapping[int(c)]


# ----------------------------------------------------------------------------
#   Preprocessing transform for datasets (mirrors `preprocess` from original)
# ----------------------------------------------------------------------------

preprocess = transforms.Compose(
    [
        audio_to_mel_spectrogram_image,
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# ----------------------------------------------------------------------------
#   RangeSampler ‑ unchanged
# ----------------------------------------------------------------------------

class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r) 