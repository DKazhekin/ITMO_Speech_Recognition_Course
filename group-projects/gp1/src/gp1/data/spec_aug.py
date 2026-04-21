"""SpecAugment for the GP1 ASR pipeline (CONTRACTS.md §4).

Applies frequency and time masking to log-mel spectrograms.
No time-warp (explicit non-goal per project plan).

Reference:
  Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method
  for Automatic Speech Recognition." https://arxiv.org/abs/1904.08779

Design:
  - Module-style (nn.Module) so it participates in .train() / .eval().
  - In eval mode: identity (no masking applied).
  - Length-aware: time masks are clamped to the valid length of each
    sample so padding frames are never masked beyond their actual end.
  - Masking = set to zero (standard SpecAugment).
  - Deterministic when a seed is passed to the constructor.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Default parameters from CONTRACTS.md §1 AugConfig.
_DEFAULT_FREQ_MASK_PARAM: int = 15
_DEFAULT_NUM_FREQ_MASKS: int = 2
_DEFAULT_TIME_MASK_PARAM: int = 25
_DEFAULT_NUM_TIME_MASKS: int = 5
_DEFAULT_TIME_MASK_MAX_RATIO: float = 0.05


def _sample_uniform_int(low: int, high: int, generator: torch.Generator) -> int:
    """Draw a uniform int in [low, high] using an explicit torch Generator."""
    if low >= high:
        return low
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


class SpecAugmenter(nn.Module):
    """Apply SpecAugment (freq-mask + time-mask) to a batch of mel spectrograms.

    No time-warp is applied (explicit non-goal for GP1).

    Args:
        freq_mask_param: Maximum frequency mask width F (default 15).
        num_freq_masks: Number of frequency masks per sample (default 2).
        time_mask_param: Maximum time mask width T (default 25).
        num_time_masks: Number of time masks per sample (default 5).
        time_mask_max_ratio: Max fraction of total time that can be masked
            in a single mask (default 0.05, per CONTRACTS.md §1).
        seed: Optional integer seed for deterministic masking.
    """

    def __init__(
        self,
        freq_mask_param: int = _DEFAULT_FREQ_MASK_PARAM,
        num_freq_masks: int = _DEFAULT_NUM_FREQ_MASKS,
        time_mask_param: int = _DEFAULT_TIME_MASK_PARAM,
        num_time_masks: int = _DEFAULT_NUM_TIME_MASKS,
        time_mask_max_ratio: float = _DEFAULT_TIME_MASK_MAX_RATIO,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_freq_masks = num_freq_masks
        self.time_mask_param = time_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_max_ratio = time_mask_max_ratio

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpecAugment in training mode; identity in eval mode.

        Args:
            mel: Log-mel spectrogram batch, shape [B, n_mels, T], float32.
            lengths: Valid frame counts per sample, shape [B], int64.

        Returns:
            Masked spectrogram batch, same shape as *mel*.
        """
        assert mel.ndim == 3, f"Expected [B, n_mels, T], got {mel.shape}"
        assert lengths.ndim == 1 and lengths.shape[0] == mel.shape[0], (
            f"lengths shape {lengths.shape} does not match batch dim {mel.shape[0]}"
        )

        if not self.training:
            return mel

        out = mel.clone()
        batch_size, n_mels, _ = out.shape

        for b in range(batch_size):
            valid_t = int(lengths[b].item())

            # --- Frequency masking ---
            for _ in range(self.num_freq_masks):
                f = _sample_uniform_int(0, self.freq_mask_param, self._generator)
                if f == 0 or n_mels <= 1:
                    continue
                f0 = _sample_uniform_int(0, n_mels - f, self._generator)
                out[b, f0 : f0 + f, :] = 0.0

            # --- Time masking (length-aware) ---
            max_t_mask = max(
                0,
                min(
                    self.time_mask_param,
                    int(valid_t * self.time_mask_max_ratio),
                    valid_t - 1,
                ),
            )
            for _ in range(self.num_time_masks):
                t = _sample_uniform_int(0, max_t_mask, self._generator)
                if t == 0 or valid_t <= 1:
                    continue
                t0 = _sample_uniform_int(0, valid_t - t, self._generator)
                out[b, :, t0 : t0 + t] = 0.0

        return out
