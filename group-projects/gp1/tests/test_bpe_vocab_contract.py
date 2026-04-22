"""Tests for BPE vocab-size contract between training and inference (C2 fix).

These tests document and enforce the contract:
    BPEVocab.vocab_size == sp.get_piece_size() + 1
and verify that inference derives vocab_size from the vocab object (dynamic),
not from the raw YAML config value (which is an off-by-one: 256 vs 257).

Root cause of C2
----------------
scripts/train.py builds FastConformerBPE with vocab_size = BPEVocab.vocab_size
= sp.get_piece_size() + 1 = 257 (for a 256-piece SP model).
inference.py:242 reads cfg["model"].get("vocab_size", 35) = 256 from YAML.
Checkpoint head is [257, d_model]; inference-built model has [256, d_model].
model.load_state_dict(state_dict) raises RuntimeError: size mismatch.

Fix: inference derives vocab_size from vocab.vocab_size, not from config.

References
----------
- PyTorch load_state_dict:
  https://pytorch.org/docs/stable/generated/torch.nn.Module.load_state_dict.html
- SentencePiece get_piece_size():
  https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Module-level skip if sentencepiece not available
# ---------------------------------------------------------------------------

spm = pytest.importorskip("sentencepiece")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TARGET_SR = 16000
# A tiny SP vocab trained on synthetic Russian corpus for these tests.
TINY_SP_VOCAB_SIZE = 100  # pieces; BPEVocab.vocab_size will be 101


def _build_tiny_sp_model(tmpdir: Path) -> Path:
    """Train a tiny SentencePiece BPE model on synthetic Russian number words.

    vocab_size=TINY_SP_VOCAB_SIZE, so BPEVocab.vocab_size = TINY_SP_VOCAB_SIZE + 1.
    """
    num2words = pytest.importorskip("num2words")

    corpus_path = tmpdir / "corpus.txt"
    model_prefix = str(tmpdir / "tiny_bpe")

    lines: list[str] = []
    for n in range(1000, 1200):
        text: str = num2words.num2words(n, lang="ru")
        text = text.replace("-", " ").lower()
        lines.append(text)
    corpus_path.write_text("\n".join(lines), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=TINY_SP_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3,
    )
    return Path(model_prefix + ".model")


def _write_wav(path: Path, duration_s: float = 0.5) -> None:
    n_samples = int(TARGET_SR * duration_s)
    audio = np.random.RandomState(42).randn(n_samples).astype(np.float32)
    sf.write(str(path), audio, TARGET_SR, subtype="FLOAT")


@pytest.fixture(scope="module")
def tiny_sp_model_path(tmp_path_factory) -> Path:
    """Shared tiny SP model for the whole module."""
    tmpdir = tmp_path_factory.mktemp("bpe_contract")
    return _build_tiny_sp_model(tmpdir)


# ---------------------------------------------------------------------------
# Test 1: BPEVocab.vocab_size == sp.get_piece_size() + 1 (contract document)
# ---------------------------------------------------------------------------


def test_bpevocab_vocab_size_equals_sp_size_plus_one(tiny_sp_model_path: Path) -> None:
    """BPEVocab.vocab_size must equal sp.get_piece_size() + 1.

    This documents the contract: blank is reserved at index 0, SP pieces
    occupy 1..N, so total = N + 1.

    Prevents future drift between BPEVocab and any caller that reads
    vocab_size from the raw YAML (which stores the SP piece count, not N+1).

    Reference: https://github.com/google/sentencepiece (get_piece_size)
    """
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)

    sp = spm.SentencePieceProcessor()
    sp.load(str(tiny_sp_model_path))

    expected = sp.get_piece_size() + 1
    assert vocab.vocab_size == expected, (
        f"BPEVocab.vocab_size={vocab.vocab_size} != "
        f"sp.get_piece_size()+1={expected}. "
        "The blank-reservation shift is broken."
    )


# ---------------------------------------------------------------------------
# Test 2: inference builds model head matching vocab object, not YAML value
# ---------------------------------------------------------------------------


def test_inference_bpe_vocab_size_matches_vocab_object(
    tiny_sp_model_path: Path, tmp_path: Path
) -> None:
    """Inference must build model with vocab_size from vocab object, not config.

    Arrange: create a FastConformerBPE checkpoint with head shape
             [vocab.vocab_size, d_model] = [101, d_model].
             Config says vocab_size: 100 (the wrong/YAML value).
    Act:     call _load_model with the vocab-derived override.
    Assert:  model head shape matches [101, d_model], not [100, d_model].

    This tests the C2 fix: vocab_size kwarg in _load_model overrides cfg value.
    """
    from gp1.models.fast_conformer_bpe import FastConformerBPE
    from gp1.submit.inference import _load_model
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    true_vocab_size = vocab.vocab_size  # e.g. 101

    d_model = 96

    # Build a model with the TRUE vocab_size (as train.py does).
    train_model = FastConformerBPE(
        vocab_size=true_vocab_size,
        d_model=d_model,
        n_blocks=2,  # tiny — fast to instantiate
        n_heads=4,
        ff_ratio=4,
        conv_kernel=9,
        dropout=0.0,
        subsample_factor=4,
    )
    ckpt_path = tmp_path / "bpe_model.pt"
    torch.save({"model": train_model.state_dict()}, ckpt_path)

    # Config has the WRONG (YAML) value — one fewer than the true vocab_size.
    cfg = {
        "model": {
            "name": "fast_conformer_bpe",
            "d_model": d_model,
            "n_blocks": 2,
            "n_heads": 4,
            "ff_ratio": 4,
            "conv_kernel": 9,
            "dropout": 0.0,
            "subsample_factor": 4,
            "vocab_size": true_vocab_size - 1,  # deliberately wrong (YAML value)
        }
    }

    device = torch.device("cpu")

    # --- Before fix: _load_model reads cfg vocab_size → shape mismatch crash ---
    # --- After fix:  _load_model accepts vocab_size kwarg → correct shape ------

    # The fix passes vocab_size as an explicit kwarg.  This test asserts that
    # when the kwarg is provided, the model is built with head shape [101, d_model].
    loaded_model = _load_model(
        ckpt_path,
        cfg,
        device,
        vocab_size=true_vocab_size,  # kwarg introduced by the C2 fix
    )

    # Verify the head (last linear layer) has the correct output dimension.
    head_weight = None
    for name, param in loaded_model.named_parameters():
        if "head" in name and "weight" in name and param.ndim == 2:
            head_weight = param
            break

    assert head_weight is not None, (
        "Could not find head weight in FastConformerBPE. "
        "Check the parameter name contains 'head'."
    )
    assert head_weight.shape[0] == true_vocab_size, (
        f"Head output dim={head_weight.shape[0]}, expected {true_vocab_size}. "
        "vocab_size kwarg not honoured by _load_model."
    )


# ---------------------------------------------------------------------------
# Test 3: end-to-end — train 1 step, save ckpt, run_inference without mismatch
# ---------------------------------------------------------------------------


def test_inference_loads_bpe_ckpt_without_shape_mismatch(
    tiny_sp_model_path: Path, tmp_path: Path
) -> None:
    """Full e2e: build model with vocab.vocab_size, save ckpt, run_inference.

    Ensures run_inference calls _build_vocab before _load_model and passes
    vocab.vocab_size into _load_model so load_state_dict never sees a shape
    mismatch.

    Reference:
    https://pytorch.org/docs/stable/generated/torch.nn.Module.load_state_dict.html
    """
    from gp1.models.fast_conformer_bpe import FastConformerBPE
    from gp1.submit.inference import InferenceConfig, run_inference
    from gp1.text.vocab_bpe import BPEVocab
    from gp1.types import ManifestRecord

    vocab = BPEVocab(tiny_sp_model_path)
    true_vocab_size = vocab.vocab_size  # e.g. 101

    d_model = 96

    # Build model with TRUE vocab_size (simulating what train.py does).
    model = FastConformerBPE(
        vocab_size=true_vocab_size,
        d_model=d_model,
        n_blocks=2,
        n_heads=4,
        ff_ratio=4,
        conv_kernel=9,
        dropout=0.0,
        subsample_factor=4,
    )

    ckpt_path = tmp_path / "e2e_bpe_model.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    # Write config YAML with the WRONG (off-by-one) vocab_size, as in the bug.
    wrong_yaml_vocab_size = (
        true_vocab_size - 1
    )  # simulates configs/fast_conformer_bpe.yaml
    bpe_model_path = str(tiny_sp_model_path)
    cfg_yaml = (
        "model:\n"
        "  name: fast_conformer_bpe\n"
        f"  d_model: {d_model}\n"
        "  n_blocks: 2\n"
        "  n_heads: 4\n"
        "  ff_ratio: 4\n"
        "  conv_kernel: 9\n"
        "  dropout: 0.0\n"
        "  subsample_factor: 4\n"
        f"  vocab_size: {wrong_yaml_vocab_size}\n"
        "audio:\n"
        "  n_fft: 512\n"
        "  samplerate: 16000\n"
        "  hop_length: 160\n"
        "  win_length: 400\n"
        "  n_mels: 80\n"
        "text:\n"
        "  vocab_type: bpe\n"
        f"  bpe_model_path: {bpe_model_path}\n"
    )
    cfg_path = tmp_path / "bpe_config.yaml"
    cfg_path.write_text(cfg_yaml)

    # Write a minimal WAV file.
    wav_path = tmp_path / "spk_A_001.wav"
    _write_wav(wav_path)

    manifest = [
        ManifestRecord(
            audio_path=wav_path,
            transcription="123456",
            spk_id="spk_A",
            gender="male",
            ext="wav",
            samplerate=TARGET_SR,
        )
    ]

    inference_cfg = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=1,
        device="cpu",
    )

    # This must NOT raise RuntimeError: size mismatch for ...
    try:
        result = run_inference(manifest, inference_cfg)
    except RuntimeError as exc:
        if "size mismatch" in str(exc).lower() or "size mismatch" in str(exc):
            pytest.fail(
                f"BPE vocab size mismatch on load_state_dict — C2 not fixed: {exc}"
            )
        raise  # other RuntimeErrors are unexpected

    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 4: config vocab_size ignored for BPE — dynamic derivation wins
# ---------------------------------------------------------------------------


def test_inference_config_vocab_size_ignored_for_bpe(
    tiny_sp_model_path: Path, tmp_path: Path
) -> None:
    """Even if cfg['model']['vocab_size'] = 1000, inference uses vocab.vocab_size.

    This guards against the temptation to 'fix' C2 by hardcoding the correct
    number in the YAML — the dynamic derivation must always supersede the config.
    """
    from gp1.models.fast_conformer_bpe import FastConformerBPE
    from gp1.submit.inference import _load_model
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    true_vocab_size = vocab.vocab_size  # e.g. 101

    d_model = 96

    # Build and save model with the true vocab_size.
    model = FastConformerBPE(
        vocab_size=true_vocab_size,
        d_model=d_model,
        n_blocks=2,
        n_heads=4,
        ff_ratio=4,
        conv_kernel=9,
        dropout=0.0,
        subsample_factor=4,
    )
    ckpt_path = tmp_path / "bpe_wrong_cfg.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    # cfg has a wildly wrong vocab_size — simulates future config drift.
    cfg = {
        "model": {
            "name": "fast_conformer_bpe",
            "d_model": d_model,
            "n_blocks": 2,
            "n_heads": 4,
            "ff_ratio": 4,
            "conv_kernel": 9,
            "dropout": 0.0,
            "subsample_factor": 4,
            "vocab_size": 1000,  # deliberately absurd wrong value
        }
    }

    device = torch.device("cpu")

    # Passing vocab_size=true_vocab_size must override the cfg value.
    loaded_model = _load_model(
        ckpt_path,
        cfg,
        device,
        vocab_size=true_vocab_size,  # dynamic override — C2 fix
    )

    # Verify load_state_dict succeeded (no RuntimeError was raised above).
    # Also check head output dimension.
    head_weight = None
    for name, param in loaded_model.named_parameters():
        if "head" in name and "weight" in name and param.ndim == 2:
            head_weight = param
            break

    assert head_weight is not None, "Could not find head weight in model."
    assert head_weight.shape[0] == true_vocab_size, (
        f"Head output dim={head_weight.shape[0]}, expected {true_vocab_size}. "
        "Dynamic vocab_size override did not supersede cfg value of 1000."
    )
