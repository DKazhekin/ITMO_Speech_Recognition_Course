"""Log mel filter-bank frontend for GP1.

Ported from ``assignments/assignment1/melbanks.py`` and adjusted per
``CONTRACTS.md §3``:

- default ``n_fft`` raised from 400 to 512 (power-of-two FFT for faster cuFFT),
- ``win_length`` now decoupled from ``n_fft`` and defaults to 400 (25 ms
  at 16 kHz). The Hann window buffer has shape ``(win_length,)``, matching
  torchaudio/Kaldi/ESPnet convention.

References:
- torchaudio melscale_fbanks: https://pytorch.org/audio/stable/generated/torchaudio.functional.melscale_fbanks.html
- Kaldi MFCC frontend (win_length != n_fft): http://kaldi-asr.org/doc/feat.html
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchaudio import functional as AF

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-6


class LogMelFilterBanks(nn.Module):
    """Compute log-mel spectrogram: ``[B, T] -> [B, n_mels, T_frames]``."""

    def __init__(
        self,
        n_fft: int = 512,
        samplerate: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min_hz: float = 0.0,
        f_max_hz: float | None = None,
        pad_mode: str = "reflect",
        power: float = 2.0,
        center: bool = True,
        norm_mel: str | None = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()

        if win_length > n_fft:
            raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

        self.n_fft = n_fft
        self.samplerate = samplerate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.pad_mode = pad_mode
        self.power = power
        self.center = center
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # Port detail: the Hann window must be sized to win_length, not n_fft.
        # The legacy implementation coupled them (win == n_fft).
        self.register_buffer("window", torch.hann_window(self.win_length))
        self.register_buffer("mel_fbanks", self._init_melscale_fbanks())

    def _init_melscale_fbanks(self) -> torch.Tensor:
        f_max = self.f_max_hz if self.f_max_hz is not None else self.samplerate / 2
        return AF.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min_hz,
            f_max=f_max,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,  # type: ignore[arg-type]
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-mel features.

        Args:
            x: ``[B, T_audio]`` float32, 16 kHz waveform.

        Returns:
            ``[B, n_mels, T_frames]`` log-mel spectrogram.
        """
        if x.ndim != 2:
            raise ValueError(
                f"LogMelFilterBanks expects [B, T] input, got shape {tuple(x.shape)}"
            )

        spec = self._spectrogram(x)  # [B, F, T']
        power_spectrum = torch.pow(spec.abs(), self.power)
        # mel_fbanks: [F, n_mels] -> transpose to [n_mels, F] for matmul.
        mel_matrix = self.mel_fbanks.T  # [n_mels, F]
        mel_spec = torch.matmul(mel_matrix, power_spectrum)  # type: ignore[arg-type]  # [B, n_mels, T']
        return (mel_spec + _LOG_EPS).log()
