"""Tests for KenLMWrapper (lm.py).

All tests mock the kenlm library entirely — no binary files required.
Tests skip cleanly if kenlm is not installed in the environment.

References:
  - CONTRACTS.md §7 — KenLMWrapper signature
  - https://github.com/kpu/kenlm — Model.score(text, bos, eos) returns log10 prob
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip whole module if kenlm is not importable, so CI doesn't fail
# on machines without the native extension.
kenlm = pytest.importorskip("kenlm", reason="kenlm not installed; skipping LM tests")

from gp1.decoding.lm import KenLMWrapper  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# Test 1: FileNotFoundError on non-existent path
# ---------------------------------------------------------------------------


def test_kenlm_wrapper_raises_file_not_found_when_binary_missing(
    tmp_path: Path,
) -> None:
    """KenLMWrapper must raise FileNotFoundError with the path in the message."""
    # Arrange
    nonexistent = tmp_path / "nonexistent_model.bin"

    # Act + Assert
    with pytest.raises(FileNotFoundError) as exc_info:
        KenLMWrapper(nonexistent)

    assert str(nonexistent) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: score delegates to kenlm.Model with default bos/eos
# ---------------------------------------------------------------------------


def test_kenlm_wrapper_delegates_score_to_kenlm_model(tmp_path: Path) -> None:
    """score() must proxy to kenlm.Model.score with bos=True, eos=True."""
    # Arrange
    fake_bin = tmp_path / "lm.bin"
    fake_bin.touch()

    mock_model = MagicMock()
    mock_model.score.return_value = -3.14

    with patch("gp1.decoding.lm.kenlm.Model", return_value=mock_model):
        wrapper = KenLMWrapper(fake_bin)

        # Act
        result = wrapper.score("тысяча один")

    # Assert
    mock_model.score.assert_called_once_with("тысяча один", bos=True, eos=True)
    assert result == pytest.approx(-3.14)


# ---------------------------------------------------------------------------
# Test 3: score forwards custom bos/eos flags
# ---------------------------------------------------------------------------


def test_kenlm_wrapper_score_passes_bos_eos_flags(tmp_path: Path) -> None:
    """score(bos=False, eos=False) must pass those exact flags to kenlm."""
    # Arrange
    fake_bin = tmp_path / "lm.bin"
    fake_bin.touch()

    mock_model = MagicMock()
    mock_model.score.return_value = -1.0

    with patch("gp1.decoding.lm.kenlm.Model", return_value=mock_model):
        wrapper = KenLMWrapper(fake_bin)

        # Act
        wrapper.score("пять тысяч", bos=False, eos=False)

    # Assert
    mock_model.score.assert_called_once_with("пять тысяч", bos=False, eos=False)
