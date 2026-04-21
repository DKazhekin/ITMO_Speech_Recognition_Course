"""Tests for BeamSearchDecoder (beam_pyctc.py).

Tests skip cleanly if pyctcdecode is not installed.
No KenLM binary is required — pass kenlm_path=None for pure hard-lexicon mode.

References:
  - CONTRACTS.md §7 — BeamSearchDecoder / BeamSearchConfig signatures
  - pyctcdecode label convention: labels[0]="" (blank), labels[1]=" " (space),
    labels[2..] = Russian letters
"""

from __future__ import annotations

import pytest
import torch

pyctcdecode = pytest.importorskip(
    "pyctcdecode", reason="pyctcdecode not installed; skipping beam tests"
)

from gp1.text.vocab import CharVocab, RUSSIAN_ALPHABET_LOWER  # noqa: E402
from gp1.decoding.beam_pyctc import (  # noqa: E402
    BeamSearchConfig,
    BeamSearchDecoder,
)


@pytest.fixture()
def vocab() -> CharVocab:
    return CharVocab()


@pytest.fixture()
def default_config() -> BeamSearchConfig:
    return BeamSearchConfig()


# ---------------------------------------------------------------------------
# Test 1: labels list built correctly from CharVocab
# ---------------------------------------------------------------------------


def test_beam_decoder_labels_built_from_char_vocab(
    vocab: CharVocab, default_config: BeamSearchConfig
) -> None:
    """labels[0]='' (blank), labels[1]=' ', labels[2]='а'."""
    # Arrange + Act
    decoder = BeamSearchDecoder(
        vocab=vocab,
        kenlm_path=None,
        unigrams=["тысяча", "тысячи", "тысяч"],
        config=default_config,
    )

    # Assert — inspect the stored labels
    assert decoder.labels[0] == ""
    assert decoder.labels[1] == " "
    assert decoder.labels[2] == "а"
    # All Russian letters must appear in order
    for idx, ch in enumerate(RUSSIAN_ALPHABET_LOWER):
        assert decoder.labels[idx + 2] == ch


# ---------------------------------------------------------------------------
# Test 2: decode_batch without KenLM returns plausible string list
# ---------------------------------------------------------------------------


def test_beam_decoder_without_kenlm_returns_plausible_string(
    vocab: CharVocab, default_config: BeamSearchConfig
) -> None:
    """decode_batch must return a list[str] of length B with no crash."""
    # Arrange
    torch.manual_seed(0)
    B, T, V = 2, 20, vocab.vocab_size
    log_probs = torch.randn(B, T, V)
    output_lengths = torch.full((B,), T, dtype=torch.int64)

    decoder = BeamSearchDecoder(
        vocab=vocab,
        kenlm_path=None,
        unigrams=["тысяча", "тысячи", "тысяч"],
        config=default_config,
    )

    # Act
    results = decoder.decode_batch(log_probs, output_lengths)

    # Assert
    assert isinstance(results, list)
    assert len(results) == B
    for s in results:
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Test 3: output_lengths truncation (padding ignored)
# ---------------------------------------------------------------------------


def test_beam_decoder_decode_batch_respects_output_lengths(
    vocab: CharVocab, default_config: BeamSearchConfig
) -> None:
    """Padding timesteps beyond output_lengths[i] must not affect the output."""
    # Arrange: two identical frames followed by wildly different padding
    torch.manual_seed(1)
    T_real = 8
    T_pad = 30
    V = vocab.vocab_size

    # Build deterministic real frames
    real_frames = torch.randn(1, T_real, V)
    # Padding: all mass on token id=5 (letter 'в') — a clearly different signal
    pad_frames = torch.full((1, T_pad - T_real, V), fill_value=-1e9)
    pad_frames[:, :, 5] = 0.0

    log_probs_with_pad = torch.cat([real_frames, pad_frames], dim=1)
    log_probs_no_pad = real_frames.clone()

    output_lengths_full = torch.tensor([T_real], dtype=torch.int64)

    decoder = BeamSearchDecoder(
        vocab=vocab,
        kenlm_path=None,
        unigrams=["тысяча", "тысячи", "тысяч"],
        config=default_config,
    )

    # Act: decode padded tensor with output_length=T_real; decode unpadded tensor
    result_padded = decoder.decode_batch(log_probs_with_pad, output_lengths_full)
    result_clean = decoder.decode_batch(log_probs_no_pad, output_lengths_full)

    # Assert: both must give the same string since padding is ignored
    assert result_padded == result_clean


# ---------------------------------------------------------------------------
# Test 4: hotword weight steers decoding toward hotword
# ---------------------------------------------------------------------------


def test_beam_decoder_applies_hotword_weight(
    vocab: CharVocab,
) -> None:
    """High hotword_weight on 'тысяча' should steer the decoded output.

    We craft a log_prob sequence where 'тысяча' is a plausible hypothesis
    by giving high probability to its character sequence, then confirm
    the decoder returns it when hotword_weight is large.

    Accept: the test may be fragile with random logits; we use a fixed seed
    and a deliberately strong signal.
    """
    # Arrange
    torch.manual_seed(0)

    # Build a [1, T, V] tensor that strongly scores the characters т-ы-с-я-ч-а
    # followed by blanks.
    word = "тысяча"
    # Encode through vocab to get ids
    ids = vocab.encode(word)  # e.g. [21, 31, 20, 32, 26, 2] (depends on alphabet order)
    T = len(ids) + 4  # add a few blank frames at end
    V = vocab.vocab_size

    lp = torch.full((T, V), fill_value=-20.0)
    for t, tok_id in enumerate(ids):
        lp[t, tok_id] = 0.0  # mass entirely on the letter
    for t in range(len(ids), T):
        lp[t, 0] = 0.0  # blank frames

    log_probs = lp.unsqueeze(0)
    output_lengths = torch.tensor([T], dtype=torch.int64)

    config_hot = BeamSearchConfig(
        beam_width=10,
        hotwords=("тысяча",),
        hotword_weight=20.0,
    )
    decoder = BeamSearchDecoder(
        vocab=vocab,
        kenlm_path=None,
        unigrams=list(vocab.encode.__doc__ and ["тысяча"] or ["тысяча"]),
        config=config_hot,
    )

    # Act
    results = decoder.decode_batch(log_probs, output_lengths)

    # Assert: 'тысяча' should appear in the decoded string
    assert "тысяча" in results[0], (
        f"Expected 'тысяча' in decoded output, got: {results[0]!r}"
    )
