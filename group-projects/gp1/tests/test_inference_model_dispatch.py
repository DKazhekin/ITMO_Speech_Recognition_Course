"""Tests for model registry and vocab dispatch in inference.py.

RED phase: these tests must fail before the implementation is in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — minimal fake checkpoint on disk
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(tmp_path: Path, state_dict: dict) -> Path:
    """Write a minimal checkpoint file acceptable by _load_model."""
    ckpt = tmp_path / "fake.pt"
    torch.save({"model": state_dict}, ckpt)
    return ckpt


def _minimal_state_for(model) -> dict:
    """Return the model's own state_dict so load_state_dict succeeds."""
    return model.state_dict()


# ---------------------------------------------------------------------------
# Import helpers — pulled inside tests to avoid import errors at collection
# ---------------------------------------------------------------------------


def _get_dispatch_symbols():
    """Import dispatch symbols; raises ImportError if not yet implemented."""
    from gp1.submit.inference import _build_vocab, _load_model  # noqa: PLC0415

    return _load_model, _build_vocab


# ---------------------------------------------------------------------------
# _load_model dispatch tests
# ---------------------------------------------------------------------------


class TestLoadModelDispatch:
    """Verify _load_model dispatches to the correct class for each model name."""

    def _cfg(self, name: str, **overrides) -> dict[str, Any]:
        base: dict[str, Any] = {"model": {"name": name, "vocab_size": 35}}
        base["model"].update(overrides)
        return base

    # -- QuartzNet10x4 (regression guard) ------------------------------------

    def test_quartznet_10x4_returns_quartznet_instance(self, tmp_path):
        _load_model, _ = _get_dispatch_symbols()
        from gp1.models.quartznet import QuartzNet10x4

        cfg = self._cfg("quartznet_10x4", d_model=256, dropout=0.1, subsample_factor=2)
        ref_model = QuartzNet10x4(
            vocab_size=35, d_model=256, dropout=0.1, subsample_factor=2
        )
        ckpt = _make_fake_checkpoint(tmp_path, _minimal_state_for(ref_model))

        model = _load_model(ckpt, cfg, torch.device("cpu"))
        assert isinstance(model, QuartzNet10x4)

    # -- CRDNN ---------------------------------------------------------------

    def test_crdnn_returns_crdnn_instance(self, tmp_path):
        _load_model, _ = _get_dispatch_symbols()
        from gp1.models.crdnn import CRDNN

        cfg = self._cfg(
            "crdnn",
            d_cnn=64,
            rnn_hidden=256,
            rnn_layers=2,
            dropout=0.15,
            subsample_factor=1,
        )
        ref_model = CRDNN(
            vocab_size=35,
            d_cnn=64,
            rnn_hidden=256,
            rnn_layers=2,
            dropout=0.15,
            subsample_factor=1,
        )
        ckpt = _make_fake_checkpoint(tmp_path, _minimal_state_for(ref_model))

        model = _load_model(ckpt, cfg, torch.device("cpu"))
        assert isinstance(model, CRDNN)

    # -- EfficientConformer --------------------------------------------------

    def test_efficient_conformer_returns_efficient_conformer_instance(self, tmp_path):
        _load_model, _ = _get_dispatch_symbols()
        from gp1.models.efficient_conformer import EfficientConformer

        cfg = self._cfg(
            "efficient_conformer",
            d_model_stages=[96, 128, 128],
            n_blocks_per_stage=[4, 4, 4],
            n_heads=4,
            ff_ratio=4,
            conv_kernel=15,
            dropout=0.1,
        )
        ref_model = EfficientConformer(
            vocab_size=35,
            d_model_stages=(96, 128, 128),
            n_blocks_per_stage=(4, 4, 4),
            n_heads=4,
            ff_ratio=4,
            conv_kernel=15,
            dropout=0.1,
        )
        ckpt = _make_fake_checkpoint(tmp_path, _minimal_state_for(ref_model))

        model = _load_model(ckpt, cfg, torch.device("cpu"))
        assert isinstance(model, EfficientConformer)

    # -- FastConformerBPE ----------------------------------------------------

    def test_fast_conformer_bpe_returns_fast_conformer_bpe_instance(self, tmp_path):
        _load_model, _ = _get_dispatch_symbols()
        from gp1.models.fast_conformer_bpe import FastConformerBPE

        cfg = self._cfg(
            "fast_conformer_bpe",
            d_model=96,
            n_blocks=16,
            n_heads=4,
            ff_ratio=4,
            conv_kernel=9,
            dropout=0.1,
            subsample_factor=4,
        )
        ref_model = FastConformerBPE(
            vocab_size=35,
            d_model=96,
            n_blocks=16,
            n_heads=4,
            ff_ratio=4,
            conv_kernel=9,
            dropout=0.1,
            subsample_factor=4,
        )
        ckpt = _make_fake_checkpoint(tmp_path, _minimal_state_for(ref_model))

        model = _load_model(ckpt, cfg, torch.device("cpu"))
        assert isinstance(model, FastConformerBPE)

    # -- Unknown name raises ValueError -------------------------------------

    def test_unknown_model_name_raises_value_error(self, tmp_path):
        _load_model, _ = _get_dispatch_symbols()

        cfg = {"model": {"name": "totally_unknown_arch", "vocab_size": 35}}
        ckpt = _make_fake_checkpoint(tmp_path, {})

        with pytest.raises(ValueError, match="totally_unknown_arch"):
            _load_model(ckpt, cfg, torch.device("cpu"))

    # -- Return type is nn.Module -------------------------------------------

    def test_load_model_returns_nn_module(self, tmp_path):
        import torch.nn as nn

        _load_model, _ = _get_dispatch_symbols()
        from gp1.models.quartznet import QuartzNet10x4

        cfg = {"model": {"name": "quartznet_10x4", "vocab_size": 35}}
        ref_model = QuartzNet10x4(vocab_size=35)
        ckpt = _make_fake_checkpoint(tmp_path, _minimal_state_for(ref_model))

        model = _load_model(ckpt, cfg, torch.device("cpu"))
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# _build_vocab dispatch tests
# ---------------------------------------------------------------------------


class TestBuildVocab:
    """Verify _build_vocab dispatches on cfg['text']['vocab_type']."""

    def test_no_text_section_returns_char_vocab(self):
        _, _build_vocab = _get_dispatch_symbols()
        from gp1.text.vocab import CharVocab

        vocab = _build_vocab({}, config_path=Path("/fake/config.yaml"))
        assert isinstance(vocab, CharVocab)

    def test_vocab_type_char_returns_char_vocab(self):
        _, _build_vocab = _get_dispatch_symbols()
        from gp1.text.vocab import CharVocab

        cfg = {"text": {"vocab_type": "char"}}
        vocab = _build_vocab(cfg, config_path=Path("/fake/config.yaml"))
        assert isinstance(vocab, CharVocab)

    def test_vocab_type_bpe_returns_bpe_vocab(self, tmp_path):
        _, _build_vocab = _get_dispatch_symbols()

        # Create a fake .model file so BPEVocab.__init__ finds it.
        bpe_model = tmp_path / "fake.model"
        bpe_model.write_bytes(b"fake sentencepiece model content")

        cfg = {"text": {"vocab_type": "bpe", "bpe_model_path": str(bpe_model)}}

        # BPEVocab internally calls sentencepiece; mock the heavy SP load.
        mock_sp = MagicMock()
        mock_sp.get_piece_size.return_value = 255

        with patch("gp1.text.vocab_bpe._import_sentencepiece") as mock_import:
            sp_module = MagicMock()
            sp_module.SentencePieceProcessor.return_value = mock_sp
            mock_import.return_value = sp_module

            from gp1.text.vocab_bpe import BPEVocab

            vocab = _build_vocab(cfg, config_path=Path("/fake/config.yaml"))
            assert isinstance(vocab, BPEVocab)

    def test_bpe_relative_path_resolved_against_config_dir(self, tmp_path):
        _, _build_vocab = _get_dispatch_symbols()

        # Place bpe model file in a subdir; config lives at tmp_path/config.yaml
        bpe_dir = tmp_path / "bpe"
        bpe_dir.mkdir()
        bpe_model = bpe_dir / "model.model"
        bpe_model.write_bytes(b"fake")

        cfg = {"text": {"vocab_type": "bpe", "bpe_model_path": "bpe/model.model"}}
        config_path = tmp_path / "config.yaml"

        mock_sp = MagicMock()
        mock_sp.get_piece_size.return_value = 255

        with patch("gp1.text.vocab_bpe._import_sentencepiece") as mock_import:
            sp_module = MagicMock()
            sp_module.SentencePieceProcessor.return_value = mock_sp
            mock_import.return_value = sp_module

            from gp1.text.vocab_bpe import BPEVocab

            vocab = _build_vocab(cfg, config_path=config_path)
            assert isinstance(vocab, BPEVocab)

    def test_bpe_missing_model_file_raises_clear_error(self, tmp_path):
        _, _build_vocab = _get_dispatch_symbols()

        cfg = {"text": {"vocab_type": "bpe", "bpe_model_path": "nonexistent.model"}}
        config_path = tmp_path / "config.yaml"

        with pytest.raises((FileNotFoundError, ValueError)):
            _build_vocab(cfg, config_path=config_path)
