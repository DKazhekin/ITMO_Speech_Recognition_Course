"""Tests for AudioAugmenter (CONTRACTS.md §4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def wav_1s() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(16000, dtype=torch.float32)


def test_augmenter_is_deterministic_under_fixed_seed(wav_1s: torch.Tensor) -> None:
    # Arrange
    from gp1.data.audio_aug import AudioAugmenter
    from gp1.types import AugConfig

    cfg = AugConfig(seed=42, musan_root=None, rir_root=None)

    # Act
    out_a = AudioAugmenter(cfg)(wav_1s.clone())
    out_b = AudioAugmenter(cfg)(wav_1s.clone())

    # Assert
    assert out_a.shape == out_b.shape
    torch.testing.assert_close(out_a, out_b, rtol=1e-6, atol=1e-6)


def test_speed_and_pitch_are_xor(wav_1s: torch.Tensor) -> None:
    """In any single call at most one of {speed-change, pitch-shift} may apply."""
    # Arrange
    from gp1.data import audio_aug as aa
    from gp1.types import AugConfig

    speed_calls = 0
    pitch_calls = 0

    real_speed = aa._apply_speed_perturb
    real_pitch = aa._apply_pitch_shift

    def count_speed(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal speed_calls
        speed_calls += 1
        return real_speed(*args, **kwargs)

    def count_pitch(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal pitch_calls
        pitch_calls += 1
        return real_pitch(*args, **kwargs)

    aa._apply_speed_perturb = count_speed  # type: ignore[assignment]
    aa._apply_pitch_shift = count_pitch  # type: ignore[assignment]

    try:
        # Force speed always on, pitch always on — XOR in impl must pick one.
        cfg = AugConfig(
            seed=7,
            speed_prob=1.0,
            pitch_prob=1.0,
            vtlp_prob=0.0,
            gain_prob=0.0,
            noise_prob=0.0,
            rir_prob=0.0,
            musan_root=None,
            rir_root=None,
        )
        aug = aa.AudioAugmenter(cfg)

        n_calls = 50
        for _ in range(n_calls):
            aug(wav_1s.clone())

        # Each invocation calls at most one of the two.
        assert speed_calls + pitch_calls == n_calls, (
            f"speed={speed_calls} pitch={pitch_calls} total={n_calls}"
        )
        assert speed_calls > 0, "speed branch never chosen"
        assert pitch_calls > 0, "pitch branch never chosen"
    finally:
        aa._apply_speed_perturb = real_speed  # type: ignore[assignment]
        aa._apply_pitch_shift = real_pitch  # type: ignore[assignment]


def test_augmenter_preserves_float32_dtype(wav_1s: torch.Tensor) -> None:
    # Arrange
    from gp1.data.audio_aug import AudioAugmenter
    from gp1.types import AugConfig

    cfg = AugConfig(seed=0, musan_root=None, rir_root=None)
    aug = AudioAugmenter(cfg)

    # Act
    out = aug(wav_1s)

    # Assert
    assert out.dtype == torch.float32
    assert out.ndim == 1


def test_augmenter_handles_missing_musan_and_rir_paths(wav_1s: torch.Tensor) -> None:
    # Arrange
    from gp1.data.audio_aug import AudioAugmenter
    from gp1.types import AugConfig

    cfg = AugConfig(
        seed=1,
        noise_prob=1.0,  # force-on, but musan_root is None -> must skip
        rir_prob=1.0,
        musan_root=None,
        rir_root=None,
    )
    aug = AudioAugmenter(cfg)

    # Act
    out = aug(wav_1s)

    # Assert: no crash, finite output
    assert torch.isfinite(out).all()
