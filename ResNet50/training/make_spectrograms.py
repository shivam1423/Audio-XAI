#!/usr/bin/env python
"""Pre-compute log-mel spectrogram PNGs for the ESC-50 dataset.

This is optional – the `ESC50SpectrogramDataset` already generates
spectrograms on-the-fly, but pre-computing speeds up training.

Example
-------
python -m RISE_audio.training.make_spectrograms \
        --esc50_root ESC50 \
        --out_dir RISE_audio/ESC50_spectrograms
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Local imports
try:
    from ..utils import audio_to_mel_spectrogram_image
except ImportError:  # script executed directly
    from RISE_audio.utils import audio_to_mel_spectrogram_image  # type: ignore

def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute ESC-50 spectrogram images")
    p.add_argument("--esc50_root", type=str, required=True, help="Path to ESC-50 dataset root (must contain 'audio/' and 'meta/esc50.csv')")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save generated PNGs (will mirror category structure)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    p.add_argument("--sr", type=int, default=22050, help="Sample rate for librosa.load")
    p.add_argument("--n_mels", type=int, default=128, help="Number of mel bins")
    return p.parse_args()


def main():
    args = parse_args()

    esc50_root = Path(args.esc50_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(esc50_root / "meta" / "esc50.csv")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating spectrogram PNGs"):
        wav_file = esc50_root / "audio" / row["filename"]
        category = row["category"]
        # Output path mirrors class name folders for torchvision.ImageFolder compatibility
        class_dir = out_dir / category
        class_dir.mkdir(parents=True, exist_ok=True)
        png_path = class_dir / f"{wav_file.stem}.png"
        if png_path.exists() and not args.overwrite:
            continue
        try:
            img = audio_to_mel_spectrogram_image(
                str(wav_file), sr=args.sr, n_mels=args.n_mels
            )
            img.save(png_path)
        except Exception as exc:
            print(f"[WARN] Failed to process {wav_file}: {exc}")

    print(f"Finished. Spectrogram images saved under {out_dir.resolve()}")


if __name__ == "__main__":
    main() 