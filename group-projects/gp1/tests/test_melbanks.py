"""Tests for LogMelFilterBanks port (CONTRACTS.md §3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Path to the legacy reference implementation we are porting from.
_ASSIGNMENT_MELBANKS = (
    REPO_ROOT.parents[1] / "assignments" / "assignment1" / "melbanks.py"
)


def test_forward_returns_expected_shape_for_one_second_audio() -> None:
    # Arrange
    from gp1.features.melbanks import LogMelFilterBanks

    mel = LogMelFilterBanks()
    wav = torch.randn(1, 16000)

    # Act
    feats = mel(wav)

    # Assert
    assert feats.shape == (1, 80, 101)


def test_forward_accepts_batch_dimension() -> None:
    # Arrange
    from gp1.features.melbanks import LogMelFilterBanks

    mel = LogMelFilterBanks()
    wav = torch.randn(4, 16000)

    # Act
    feats = mel(wav)

    # Assert
    assert feats.shape == (4, 80, 101)


def test_default_parameters_match_contract() -> None:
    # Arrange + Act
    from gp1.features.melbanks import LogMelFilterBanks

    mel = LogMelFilterBanks()

    # Assert — CONTRACTS.md §3 defaults
    assert mel.n_fft == 512
    assert mel.hop_length == 160
    assert mel.win_length == 400
    assert mel.n_mels == 80
    assert mel.samplerate == 16000
    assert mel.window.shape == (400,), "window must be win_length, not n_fft"


def test_port_matches_legacy_impl_on_fixed_input() -> None:
    """Port with matched kwargs (n_fft=400, win_length=400) must equal legacy."""
    # Arrange
    if not _ASSIGNMENT_MELBANKS.exists():
        pytest.skip(f"legacy melbanks.py not found at {_ASSIGNMENT_MELBANKS}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_legacy_melbanks", _ASSIGNMENT_MELBANKS
    )
    assert spec is not None and spec.loader is not None
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)
    from gp1.features.melbanks import LogMelFilterBanks

    torch.manual_seed(0)
    wav = torch.randn(1, 16000)

    # Legacy has n_fft == win_length by construction.
    legacy = legacy_mod.LogMelFilterBanks(n_fft=400, hop_length=160, n_mels=80)
    ported = LogMelFilterBanks(
        n_fft=400, hop_length=160, win_length=400, n_mels=80
    )

    # Act
    legacy_out = legacy(wav)
    ported_out = ported(wav)

    # Assert
    assert legacy_out.shape == ported_out.shape
    torch.testing.assert_close(legacy_out, ported_out, rtol=1e-5, atol=1e-5)


def test_output_is_finite() -> None:
    # Arrange
    from gp1.features.melbanks import LogMelFilterBanks

    mel = LogMelFilterBanks()
    wav = torch.randn(2, 8000)

    # Act
    feats = mel(wav)

    # Assert
    assert torch.isfinite(feats).all()
