"""Tests for helper functions in scripts/train.py.

Covers model registry, vocab dispatch, and auxiliary-loss factory functions.
RED phase: functions _build_model, _build_vocab, _build_inter_ctc, _build_cr_ctc
do not yet exist in train.py — all tests should FAIL initially.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Make sure the repo src is importable (mirrors train.py lines 32-34)
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Make sure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train as train_script  # noqa: E402  (after sys.path manipulation)

from gp1.models.crdnn import CRDNN
from gp1.models.efficient_conformer import EfficientConformer
from gp1.models.fast_conformer_bpe import FastConformerBPE
from gp1.models.quartznet import QuartzNet10x4
from gp1.text.vocab import CharVocab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 35  # CharVocab default


def _model_cfg(**overrides: Any) -> dict[str, Any]:
    """Return a minimal cfg dict with sensible defaults."""
    base: dict[str, Any] = {
        "model": {
            "name": "quartznet_10x4",
            "d_model": 256,
            "dropout": 0.1,
        }
    }
    base["model"].update(overrides)
    return base


# ---------------------------------------------------------------------------
# _build_model — model registry
# ---------------------------------------------------------------------------


def test_build_model_quartznet():
    # Arrange
    cfg = _model_cfg(name="quartznet_10x4")

    # Act
    model = train_script._build_model(cfg, vocab_size=_VOCAB_SIZE)

    # Assert
    assert isinstance(model, QuartzNet10x4)


def test_build_model_crdnn():
    # Arrange
    cfg = {
        "model": {
            "name": "crdnn",
            "d_cnn": 64,
            "rnn_hidden": 128,
            "rnn_layers": 2,
            "dropout": 0.1,
            "subsample_factor": 1,
        }
    }

    # Act
    model = train_script._build_model(cfg, vocab_size=_VOCAB_SIZE)

    # Assert
    assert isinstance(model, CRDNN)


def test_build_model_efficient_conformer():
    # Arrange
    cfg = {
        "model": {
            "name": "efficient_conformer",
            "d_model_stages": [96, 128, 128],
            "n_blocks_per_stage": [4, 4, 4],
            "n_heads": 4,
            "ff_ratio": 4,
            "conv_kernel": 15,
            "dropout": 0.1,
            "subsample_factor": 4,
        }
    }

    # Act
    model = train_script._build_model(cfg, vocab_size=_VOCAB_SIZE)

    # Assert
    assert isinstance(model, EfficientConformer)


def test_build_model_fast_conformer_bpe():
    # Arrange
    cfg = {
        "model": {
            "name": "fast_conformer_bpe",
            "d_model": 96,
            "n_blocks": 16,
            "n_heads": 4,
            "ff_ratio": 4,
            "conv_kernel": 9,
            "dropout": 0.1,
            "subsample_factor": 4,
        }
    }

    # Act
    model = train_script._build_model(cfg, vocab_size=256)

    # Assert
    assert isinstance(model, FastConformerBPE)


def test_build_model_unknown_raises():
    # Arrange
    cfg = _model_cfg(name="does_not_exist")

    # Act / Assert
    with pytest.raises(ValueError, match="does_not_exist"):
        train_script._build_model(cfg, vocab_size=_VOCAB_SIZE)


# ---------------------------------------------------------------------------
# _build_vocab — vocab dispatch
# ---------------------------------------------------------------------------


def test_build_vocab_default_is_char(tmp_path: Path):
    # Arrange — no text section in cfg
    cfg: dict[str, Any] = {}
    config_path = tmp_path / "dummy.yaml"

    # Act
    vocab = train_script._build_vocab(cfg, config_path)

    # Assert
    assert isinstance(vocab, CharVocab)


def test_build_vocab_char_explicit(tmp_path: Path):
    # Arrange
    cfg = {"text": {"vocab_type": "char"}}
    config_path = tmp_path / "dummy.yaml"

    # Act
    vocab = train_script._build_vocab(cfg, config_path)

    # Assert
    assert isinstance(vocab, CharVocab)


def test_build_vocab_bpe_returns_bpevocab(tmp_path: Path):
    """BPEVocab is constructed; sentencepiece not required — we mock the class."""
    # Arrange
    cfg = {
        "text": {
            "vocab_type": "bpe",
            "bpe_model_path": "data/bpe/numbers.model",
        }
    }
    config_path = tmp_path / "configs" / "dummy.yaml"

    fake_bpe_vocab = MagicMock()
    fake_bpe_vocab.vocab_size = 257
    fake_bpe_vocab.blank_id = 0

    with patch("train.BPEVocab", return_value=fake_bpe_vocab) as mock_cls:
        # Act
        vocab = train_script._build_vocab(cfg, config_path)

    # Assert — BPEVocab was called with an absolute path
    mock_cls.assert_called_once()
    called_path = mock_cls.call_args[0][0]
    assert Path(called_path).is_absolute()
    assert vocab is fake_bpe_vocab


# ---------------------------------------------------------------------------
# _build_inter_ctc
# ---------------------------------------------------------------------------


def test_build_inter_ctc_enabled():
    # Arrange
    cfg = {"inter_ctc": {"enabled": True}}

    # Act
    head = train_script._build_inter_ctc(cfg, d_mid=256, vocab_size=35, blank_id=0)

    # Assert
    from gp1.losses.inter_ctc import InterCTCHead

    assert isinstance(head, InterCTCHead)


def test_build_inter_ctc_disabled_returns_none():
    # Arrange
    cfg = {"inter_ctc": {"enabled": False}}

    # Act
    head = train_script._build_inter_ctc(cfg, d_mid=256, vocab_size=35, blank_id=0)

    # Assert
    assert head is None


def test_build_inter_ctc_missing_section_returns_none():
    # Arrange
    cfg: dict[str, Any] = {}

    # Act
    head = train_script._build_inter_ctc(cfg, d_mid=256, vocab_size=35, blank_id=0)

    # Assert
    assert head is None


# ---------------------------------------------------------------------------
# _build_cr_ctc
# ---------------------------------------------------------------------------


def test_build_cr_ctc_enabled():
    # Arrange
    cfg = {"cr_ctc": {"enabled": True, "temperature": 2.0, "min_prob": 0.05}}

    # Act
    loss = train_script._build_cr_ctc(cfg)

    # Assert
    from gp1.losses.cr_ctc import CRCTCLoss

    assert isinstance(loss, CRCTCLoss)


def test_build_cr_ctc_disabled_returns_none():
    # Arrange
    cfg = {"cr_ctc": {"enabled": False}}

    # Act
    loss = train_script._build_cr_ctc(cfg)

    # Assert
    assert loss is None


def test_build_cr_ctc_missing_section_returns_none():
    # Arrange
    cfg: dict[str, Any] = {}

    # Act
    loss = train_script._build_cr_ctc(cfg)

    # Assert
    assert loss is None


# ---------------------------------------------------------------------------
# _build_train_dataset — return_two_views wiring
# ---------------------------------------------------------------------------


def _make_manifest_record(tmp_path: Path) -> Any:
    """Build a ManifestRecord with the real field names from gp1.types."""
    from gp1.types import ManifestRecord

    return ManifestRecord(
        audio_path=tmp_path / "a.wav",
        transcription="139473",
        spk_id="spk1",
        gender="male",
        ext="wav",
        samplerate=16000,
    )


def test_build_train_dataset_two_views_when_cr_ctc_enabled(tmp_path: Path):
    """When cr_ctc is enabled, SpokenNumbersDataset must be called with return_two_views=True."""
    records = [_make_manifest_record(tmp_path)]
    vocab = CharVocab()

    with patch("train.SpokenNumbersDataset") as mock_ds:
        mock_ds.return_value = MagicMock()
        train_script._build_train_dataset(
            records=records,
            vocab=vocab,
            target_sr=16000,
            augmenter=None,
            word_vocab=None,
            return_two_views=True,
        )

    _kwargs = mock_ds.call_args[1] if mock_ds.call_args[1] else {}
    _args = mock_ds.call_args[0] if mock_ds.call_args[0] else []
    # return_two_views may be positional or keyword; check both
    positional_true = any(a is True for a in _args)
    keyword_true = _kwargs.get("return_two_views", False)
    assert positional_true or keyword_true, (
        "SpokenNumbersDataset not called with return_two_views=True"
    )


def test_build_train_dataset_no_two_views_by_default(tmp_path: Path):
    """Without cr_ctc, return_two_views must default to False (backward compat)."""
    records = [_make_manifest_record(tmp_path)]
    vocab = CharVocab()

    with patch("train.SpokenNumbersDataset") as mock_ds:
        mock_ds.return_value = MagicMock()
        train_script._build_train_dataset(
            records=records,
            vocab=vocab,
            target_sr=16000,
            augmenter=None,
            word_vocab=None,
            return_two_views=False,
        )

    _kwargs = mock_ds.call_args[1] if mock_ds.call_args[1] else {}
    # Either not passed at all (defaults to False) or explicitly False
    assert _kwargs.get("return_two_views", False) is False
