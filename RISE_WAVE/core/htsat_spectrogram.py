#!/usr/bin/env python
# coding: utf-8

"""HTSAT spectrogram preprocessing and inference helpers."""

import math
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

import librosa
try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
except ImportError:  # pragma: no cover
    Spectrogram = None
    LogmelFilterBank = None


class HTSATSpectrogramPreprocessor(nn.Module):
    """Replicates the official HTSAT STFT + logmel front-end."""

    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        clip_samples: int = 320000,
    ):
        super().__init__()

        if Spectrogram is None or LogmelFilterBank is None:
            raise ImportError(
                "torchlibrosa is required for HTSAT spectrogram preprocessing. "
                "Install via `pip install torchlibrosa`."
            )

        self.sample_rate = sample_rate
        self.clip_samples = clip_samples

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

    def forward(self, source: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert audio (path or waveform) into HTSAT mel spectrogram."""
        waveform = self._load_waveform(source)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.spectrogram_extractor(waveform)
        mel = self.logmel_extractor(mel)
        # Return (B, 1, mel_bins, time_frames) for compatibility with TF masks
        mel = mel.permute(0, 1, 3, 2).contiguous()
        return mel

    def _load_waveform(self, source):
        if isinstance(source, str):
            return self._load_from_path(source)
        if isinstance(source, np.ndarray):
            tensor = torch.from_numpy(source).float()
        elif isinstance(source, torch.Tensor):
            tensor = source.float().clone()
        else:
            raise ValueError("Unsupported audio source type.")

        tensor = tensor.squeeze()
        tensor = self._pad_or_trim(tensor)
        return tensor

    def _load_from_path(self, path: str) -> torch.Tensor:
        if torchaudio is not None:
            try:
                waveform, sr = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                        waveform
                    )
                waveform = waveform.squeeze(0)
                waveform = self._pad_or_trim(waveform)
                return waveform
            except Exception:
                pass

        samples, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        tensor = torch.from_numpy(samples).float()
        tensor = self._pad_or_trim(tensor)
        return tensor

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.clip_samples:
            repeats = int(math.ceil(self.clip_samples / waveform.numel()))
            waveform = waveform.repeat(repeats)
        return waveform[: self.clip_samples]


def forward_htsat_from_spectrogram(
    htsat_model: nn.Module, mel: torch.Tensor, infer_mode: bool = True
):
    """
    Forward pass that bypasses HTSAT's internal spectrogram extractor.
    Expects mel shape: (B, 1, time_steps, mel_bins)
    """
    x = mel
    x = x.transpose(1, 3)
    x = htsat_model.bn0(x)
    x = x.transpose(1, 3)

    if infer_mode:
        frame_num = x.shape[2]
        target_T = int(htsat_model.spec_size * htsat_model.freq_ratio)
        repeat_ratio = max(1, math.floor(target_T / max(1, frame_num)))
        x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
        x = htsat_model.reshape_wav2img(x)
        output_dict = htsat_model.forward_features(x)
    elif getattr(htsat_model.config, "enable_repeat_mode", False):
        output_dicts = []
        for cur_pos in range(
            0,
            (htsat_model.freq_ratio - 1) * htsat_model.spec_size + 1,
            htsat_model.spec_size,
        ):
            tx = x.clone()
            tx = htsat_model.repeat_wat2img(tx, cur_pos)
            output_dicts.append(htsat_model.forward_features(tx))
        clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(
            x.device
        )
        framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(
            x.device
        )
        for d in output_dicts:
            clipwise_output += d["clipwise_output"]
            framewise_output += d["framewise_output"]
        clipwise_output /= len(output_dicts)
        framewise_output /= len(output_dicts)
        output_dict = {
            "clipwise_output": clipwise_output,
            "framewise_output": framewise_output,
        }
    else:
        if x.shape[2] > htsat_model.freq_ratio * htsat_model.spec_size:
            overlap_size = (x.shape[2] - 1) // 4
            output_dicts = []
            crop_size = (x.shape[2] - 1) // 2
            for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                tx = htsat_model.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
                tx = htsat_model.reshape_wav2img(tx)
                output_dicts.append(htsat_model.forward_features(tx))
            clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(
                x.device
            )
            framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(
                x.device
            )
            for d in output_dicts:
                clipwise_output += d["clipwise_output"]
                framewise_output += d["framewise_output"]
            clipwise_output /= len(output_dicts)
            framewise_output /= len(output_dicts)
            output_dict = {
                "clipwise_output": clipwise_output,
                "framewise_output": framewise_output,
            }
        else:
            x = htsat_model.reshape_wav2img(x)
            output_dict = htsat_model.forward_features(x)

    return output_dict


