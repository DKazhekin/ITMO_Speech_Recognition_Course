"""Bucket P1 — pitch-shift speed regression tests.

Verifies that the GCD-rounded resample implementation of _apply_pitch_shift
completes in < 50 ms per call (was ~1000 ms with the original phase-vocoder
path due to GCD=1 polyphase filter).

References:
  - torchaudio pitch_shift algorithm (phase vocoder + resample):
    https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.pitch_shift.html
  - Polyphase resample cost with GCD=1:
    https://github.com/pytorch/audio/blob/v2.5.1/src/torchaudio/functional/functional.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------

SR = 16_000
DURATION_S = 1.0


@pytest.fixture
def wav_1s() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(int(SR * DURATION_S), dtype=torch.float32)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_pitch_shift_call_is_fast(wav_1s: torch.Tensor, rng: random.Random) -> None:
    """Each _apply_pitch_shift call must complete in < 50 ms.

    The old torchaudio.functional.pitch_shift path took ~1000 ms on CPU
    when n_steps produced an orig_freq coprime with sr (GCD=1 → ~18 000-tap
    polyphase filter). The GCD-rounded replacement should be < 1 ms.
    """
    # Arrange
    from gp1.data.audio_aug import _apply_pitch_shift

    n_steps_list = [-3, -2, -1, 1, 2, 3]
    limit_ms = 50.0

    for n_steps in n_steps_list:
        # Act
        t0 = time.perf_counter()
        _apply_pitch_shift(wav_1s.clone(), SR, (float(n_steps), float(n_steps)), rng)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Assert
        assert elapsed_ms < limit_ms, (
            f"_apply_pitch_shift(n_steps={n_steps}) took {elapsed_ms:.1f} ms, "
            f"expected < {limit_ms} ms. "
            "GCD-rounded resample fix may not be applied."
        )


def test_pitch_shift_preserves_shape_or_close(
    wav_1s: torch.Tensor, rng: random.Random
) -> None:
    """Output length must be within ±25% of input length.

    Option A trades pure pitch-only (torchaudio phase vocoder) for a
    resample-only shortcut that changes duration.  For ±3 semitones the
    theoretical length ratio is 2^(3/12) ≈ 1.189 (19%), so the tolerance
    is set to 25% to cover rounding artefacts.  For CTC training this is
    acceptable — the model never sees absolute timestamps.
    """
    # Arrange
    from gp1.data.audio_aug import _apply_pitch_shift

    n_steps_list = [-3, -2, -1, 1, 2, 3]
    tolerance = 0.25

    for n_steps in n_steps_list:
        # Act
        out = _apply_pitch_shift(
            wav_1s.clone(), SR, (float(n_steps), float(n_steps)), rng
        )

        # Assert
        ratio = out.shape[0] / wav_1s.shape[0]
        assert 1 - tolerance <= ratio <= 1 + tolerance, (
            f"n_steps={n_steps}: length ratio {ratio:.3f} outside "
            f"[{1 - tolerance:.2f}, {1 + tolerance:.2f}]"
        )


def test_pitch_shift_zero_semitones_is_noop(
    wav_1s: torch.Tensor, rng: random.Random
) -> None:
    """semitones=0 must return a waveform identical (or nearly so) to input.

    When the quantized orig_freq equals sr the function must short-circuit
    and return the input unchanged.
    """
    # Arrange
    from gp1.data.audio_aug import _apply_pitch_shift

    # Force n_steps to land in the zero bucket by using range (−0.4, +0.4).
    # After rounding, n_steps_int == 0 in the original code, so the function
    # returns the input.  Our replacement should honour the same contract.
    small_rng = random.Random(999)  # deterministic: rng.uniform(-0.4, 0.4) ~ 0
    out = _apply_pitch_shift(wav_1s.clone(), SR, (-0.4, 0.4), small_rng)

    # Assert: shape unchanged
    assert out.shape == wav_1s.shape, f"Expected shape {wav_1s.shape}, got {out.shape}"
    # Assert: values essentially identical (no resample applied)
    max_diff = (out - wav_1s).abs().max().item()
    assert max_diff < 1e-4, f"Non-trivial difference for ~zero semitones: {max_diff}"


def test_pitch_shift_changes_spectrum(wav_1s: torch.Tensor, rng: random.Random) -> None:
    """Output STFT magnitude must differ meaningfully from input at n_steps=+2.

    Ensures the function actually perturbs the signal and is not a silent no-op.
    """
    # Arrange
    from gp1.data.audio_aug import _apply_pitch_shift

    # Force exactly +2 semitones
    fixed_rng = random.Random(7)  # uniform(2.0, 2.0) always yields 2.0

    # Act
    out = _apply_pitch_shift(wav_1s.clone(), SR, (2.0, 2.0), fixed_rng)

    # Assert: L1 distance between STFT magnitudes must exceed a threshold.
    n_fft = 512
    window = torch.hann_window(n_fft)

    def magnitude(w: torch.Tensor) -> torch.Tensor:
        # Pad/trim to same length before STFT comparison
        length = min(w.shape[0], wav_1s.shape[0])
        return torch.stft(
            w[:length],
            n_fft=n_fft,
            hop_length=160,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()

    mag_in = magnitude(wav_1s)
    mag_out = magnitude(out)
    # Trim time axis to same length before comparison (output may be shorter
    # when pitching up due to Option A duration change).
    n_frames = min(mag_in.shape[1], mag_out.shape[1])
    diff = (mag_out[:, :n_frames] - mag_in[:, :n_frames]).abs().mean().item()

    assert diff > 0.005, (
        f"STFT magnitude difference {diff:.6f} is too small — "
        "pitch shift may be a no-op."
    )
