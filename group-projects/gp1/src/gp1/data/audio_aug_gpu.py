"""GPU-accelerated batch audio augmentations (VTLP, AddNoise, RIR)."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taF

log = logging.getLogger(__name__)


class GPUAudioAugmenter(nn.Module):
    """Batch audio augmentations executed on GPU after collation.

    Applies VTLP, additive noise, and RIR convolution to a padded batch
    ``[B, T_max]`` in training mode. All three augmentations are per-batch
    (single alpha / SNR / IR per forward call).

    Args:
        samplerate: Waveform sample rate in Hz.
        vtlp_prob: Probability of applying VTLP per batch.
        vtlp_alpha_range: (low, high) for uniform alpha draw.
        noise_prob: Probability of applying additive noise.
        noise_snr_db_range: (min_snr, max_snr) in dB.
        musan_root: Root directory of MUSAN noise files; ``None`` disables noise.
        rir_prob: Probability of applying RIR convolution.
        rir_root: Root directory of RIR wav files; ``None`` disables RIR.
    """

    _window: torch.Tensor

    def __init__(
        self,
        samplerate: int = 16000,
        vtlp_prob: float = 0.5,
        vtlp_alpha_range: tuple[float, float] = (0.9, 1.1),
        noise_prob: float = 0.3,
        noise_snr_db_range: tuple[float, float] = (5.0, 20.0),
        musan_root: Path | None = None,
        rir_prob: float = 0.1,
        rir_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.samplerate = samplerate
        self.vtlp_prob = float(vtlp_prob)
        self.vtlp_alpha_range = vtlp_alpha_range
        self.noise_prob = float(noise_prob)
        self.noise_snr_db_range = noise_snr_db_range
        self.rir_prob = float(rir_prob)

        # STFT config — mirrors CPU VTLP (audio_aug.py:212-214).
        self._n_fft = 512
        self._hop = 160
        self._win = 400
        self.register_buffer(
            "_window", torch.hann_window(self._win), persistent=False
        )

        self._noise_pool: list[torch.Tensor] = (
            self._load_pool(musan_root) if musan_root is not None else []
        )
        self._rir_pool: list[torch.Tensor] = (
            self._load_pool(rir_root) if rir_root is not None else []
        )

    def _load_pool(self, root: Path) -> list[torch.Tensor]:
        """Load all .wav files under *root* into a CPU tensor list (mono, pinned when possible)."""
        root = Path(root)
        if not root.exists():
            log.warning("GPUAudioAugmenter: %s does not exist", root)
            return []
        files = sorted(root.rglob("*.wav"))
        max_samples = int(0.5 * self.samplerate)
        pool: list[torch.Tensor] = []
        for p in files:
            arr, _ = sf.read(str(p), dtype="float32", always_2d=True)
            mono = arr.mean(axis=1).astype("float32", copy=False)
            mono = mono[:max_samples]
            t = torch.from_numpy(mono)
            t = t / (t.norm() + 1e-9)
            if torch.cuda.is_available():
                try:
                    t = t.pin_memory()
                except RuntimeError as exc:
                    log.warning("pin_memory failed for %s: %s", p, exc)
            pool.append(t)
        total_mb = sum(t.numel() * 4 for t in pool) / 1e6
        log.info(
            "GPUAudioAugmenter: loaded %d files from %s (%.1f MB)",
            len(pool),
            root,
            total_mb,
        )
        return pool

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Apply VTLP → RIR → AddNoise pipeline (ESPnet order) to a padded batch ``[B, T_max]``."""
        if self.vtlp_prob > 0.0 and torch.rand(1).item() < self.vtlp_prob:
            audio = self._apply_vtlp_batched(audio)
        if self._rir_pool and torch.rand(1).item() < self.rir_prob:
            audio = self._apply_rir_batched(audio)
        if self._noise_pool and torch.rand(1).item() < self.noise_prob:
            audio = self._apply_add_noise_batched(audio, audio_lengths)
        return audio

    def _apply_vtlp_batched(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply VTLP frequency warping to the whole batch via batched STFT; returns same shape."""
        low, high = self.vtlp_alpha_range
        alpha = float(torch.empty(1).uniform_(low, high).item())
        if abs(alpha - 1.0) < 1e-4:
            return audio

        stft = torch.stft(
            audio,
            n_fft=self._n_fft,
            hop_length=self._hop,
            win_length=self._win,
            window=self._window,
            return_complex=True,
        )  # [B, F, Frames]

        freq_bins = stft.shape[1]
        nyquist_bin = freq_bins - 1

        orig_bins = torch.arange(
            freq_bins, dtype=torch.float32, device=audio.device
        )
        f_boundary = nyquist_bin / max(alpha, 1.0 - (alpha - 1.0))
        warped = torch.where(
            orig_bins <= f_boundary * alpha,
            orig_bins / alpha,
            (nyquist_bin - orig_bins)
            / (nyquist_bin - f_boundary * alpha + 1e-8)
            * (nyquist_bin - f_boundary)
            + f_boundary,
        ).clamp(0.0, nyquist_bin)

        floor_idx = warped.long().clamp(0, nyquist_bin - 1)
        ceil_idx = (floor_idx + 1).clamp(0, nyquist_bin)
        frac = (warped - floor_idx.float()).view(1, -1, 1)

        stft_warped = stft[:, floor_idx] * (1.0 - frac) + stft[:, ceil_idx] * frac

        return torch.istft(
            stft_warped,
            n_fft=self._n_fft,
            hop_length=self._hop,
            win_length=self._win,
            window=self._window,
            length=audio.size(-1),
        )

    def _apply_add_noise_batched(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Mix a random noise sample at a uniform random SNR into the whole batch; returns same shape."""
        B, T = audio.shape
        idx = int(torch.randint(len(self._noise_pool), (1,)).item())
        noise = self._noise_pool[idx].to(audio.device, non_blocking=True)

        if noise.numel() < T:
            repeats = (T + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(repeats)
        noise = noise[:T].unsqueeze(0).expand(B, -1).contiguous()

        low, high = self.noise_snr_db_range
        snr_val = float(torch.empty(1).uniform_(low, high).item())
        snr = torch.full((B,), snr_val, device=audio.device, dtype=audio.dtype)

        return taF.add_noise(audio, noise, snr, lengths=audio_lengths)

    def _apply_rir_batched(self, audio: torch.Tensor) -> torch.Tensor:
        """Convolve entire batch with a randomly selected RIR; returns same shape ``[B, T]``."""
        idx = int(torch.randint(len(self._rir_pool), (1,)).item())
        ir = self._rir_pool[idx]
        ir = ir.to(audio.device, dtype=audio.dtype, non_blocking=True)
        K = ir.numel()
        kernel = ir.flip(0).view(1, 1, K)
        left = (K - 1) // 2
        right = K - 1 - left
        x = F.pad(audio.unsqueeze(1), (left, right))
        return F.conv1d(x, kernel).squeeze(1)
