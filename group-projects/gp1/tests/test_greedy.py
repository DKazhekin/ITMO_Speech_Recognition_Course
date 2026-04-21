"""Tests for greedy CTC decoding.

AAA pattern throughout. Tests verify observable behaviour (decoded strings),
not internal data structures.

References:
  - CONTRACTS.md §7 — greedy_decode signature
  - CharVocab: BLANK_ID=0, SPACE_ID=1, ids 2..34 = Russian letters
"""

from __future__ import annotations

import pytest
import torch

# conftest.py already prepends src/ to sys.path
from gp1.text.vocab import CharVocab
from gp1.decoding.greedy import greedy_decode


@pytest.fixture()
def vocab() -> CharVocab:
    return CharVocab()


def _make_one_hot_log_probs(token_sequence: list[int], vocab_size: int) -> torch.Tensor:
    """Build [T, V] log-prob tensor where each timestep is a one-hot spike.

    The chosen token gets log-prob 0.0 (probability 1.0); all others get
    a large negative value so argmax is deterministic.
    """
    T = len(token_sequence)
    lp = torch.full((T, vocab_size), fill_value=-1e9)
    for t, tok in enumerate(token_sequence):
        lp[t, tok] = 0.0
    return lp


# ---------------------------------------------------------------------------
# Test 1: duplicates are collapsed and blank stripped
# ---------------------------------------------------------------------------


def test_greedy_decode_collapses_duplicates_and_strips_blank(
    vocab: CharVocab,
) -> None:
    """Sequence [а, а, BLANK, б, б] should decode to 'аб'."""
    # Arrange
    # id 2 = 'а', id 0 = BLANK, id 3 = 'б'
    tokens = [2, 2, 0, 3, 3]
    lp = _make_one_hot_log_probs(tokens, vocab_size=vocab.vocab_size)
    log_probs = lp.unsqueeze(0)  # [1, T, V]
    output_lengths = torch.tensor([len(tokens)], dtype=torch.int64)

    # Act
    results = greedy_decode(log_probs, output_lengths, vocab)

    # Assert
    assert results == ["аб"]


# ---------------------------------------------------------------------------
# Test 2: output_lengths truncation
# ---------------------------------------------------------------------------


def test_greedy_decode_respects_output_lengths(vocab: CharVocab) -> None:
    """Trailing timesteps beyond output_lengths must be ignored."""
    # Arrange
    # CharVocab: а=2, б=3, в=4, г=5 (RUSSIAN_ALPHABET_LOWER = "абвгде...")
    # First 5 tokens: [а, BLANK, б, в, BLANK] -> 'абв'
    # Next 5 tokens (should be ignored): [г, г, г, г, г]
    id_a, id_b, id_v, id_g = 2, 3, 4, 5  # а=2, б=3, в=4, г=5
    tokens = [id_a, 0, id_b, id_v, 0, id_g, id_g, id_g, id_g, id_g]
    lp = _make_one_hot_log_probs(tokens, vocab_size=vocab.vocab_size)
    log_probs = lp.unsqueeze(0)  # [1, T=10, V]
    output_lengths = torch.tensor([5], dtype=torch.int64)  # only use first 5

    # Act
    results = greedy_decode(log_probs, output_lengths, vocab)

    # Assert
    assert results == ["абв"]


# ---------------------------------------------------------------------------
# Test 3: all blank → empty string
# ---------------------------------------------------------------------------


def test_greedy_decode_empty_when_all_blank(vocab: CharVocab) -> None:
    """When every timestep is blank, decoded string must be empty."""
    # Arrange
    tokens = [0, 0, 0, 0, 0]
    lp = _make_one_hot_log_probs(tokens, vocab_size=vocab.vocab_size)
    log_probs = lp.unsqueeze(0)  # [1, T, V]
    output_lengths = torch.tensor([len(tokens)], dtype=torch.int64)

    # Act
    results = greedy_decode(log_probs, output_lengths, vocab)

    # Assert
    assert results == [""]


# ---------------------------------------------------------------------------
# Test 4: batch shape matches B
# ---------------------------------------------------------------------------


def test_greedy_decode_batch_shape_matches_B(vocab: CharVocab) -> None:
    """decode_batch must return a list of exactly B strings."""
    # Arrange
    B = 4
    T = 6
    V = vocab.vocab_size
    torch.manual_seed(42)
    log_probs = torch.randn(B, T, V)
    output_lengths = torch.full((B,), T, dtype=torch.int64)

    # Act
    results = greedy_decode(log_probs, output_lengths, vocab)

    # Assert
    assert isinstance(results, list)
    assert len(results) == B
    for s in results:
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Test 5: greedy result equals vocab.decode of argmax ids
# ---------------------------------------------------------------------------


def test_greedy_decode_equals_vocab_decode_of_argmax(vocab: CharVocab) -> None:
    """greedy_decode output must equal vocab.decode(argmax_ids) per sample."""
    # Arrange
    torch.manual_seed(7)
    B, T, V = 3, 10, vocab.vocab_size
    log_probs = torch.randn(B, T, V)
    output_lengths = torch.tensor([10, 7, 5], dtype=torch.int64)

    # Act
    results = greedy_decode(log_probs, output_lengths, vocab)

    # Assert: manually compute expected via vocab.decode
    for i in range(B):
        length = output_lengths[i].item()
        argmax_ids = log_probs[i, :length].argmax(dim=-1).tolist()
        expected = vocab.decode(argmax_ids)
        assert results[i] == expected, (
            f"Sample {i}: got {results[i]!r}, expected {expected!r}"
        )
