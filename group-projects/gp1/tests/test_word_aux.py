"""Tests for WordAuxCTCHead — auxiliary word-level CTC.

Operates on the top encoder features and projects to the closed word vocabulary
(~31 entries = 30 Russian number words + blank).
"""
from __future__ import annotations

import pytest
import torch

from gp1.losses.word_aux import WordAuxCTCHead


WORD_VOCAB_SIZE = 31
BLANK_ID = 0
D_ENC = 256


def test_forward_returns_finite_positive_scalar() -> None:
    # Arrange
    batch, t_enc, u_word = 2, 50, 5
    enc_features = torch.randn(batch, t_enc, D_ENC)
    word_targets = torch.randint(1, WORD_VOCAB_SIZE, (batch, u_word), dtype=torch.int64)
    input_lengths = torch.full((batch,), t_enc, dtype=torch.int64)
    target_lengths = torch.full((batch,), u_word, dtype=torch.int64)
    head = WordAuxCTCHead(d_enc=D_ENC, word_vocab_size=WORD_VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(enc_features, input_lengths, word_targets, target_lengths)

    # Assert
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0


def test_projection_has_correct_shape() -> None:
    # Arrange
    head = WordAuxCTCHead(d_enc=D_ENC, word_vocab_size=WORD_VOCAB_SIZE, blank_id=BLANK_ID)

    # Act / Assert
    assert head.proj.in_features == D_ENC
    assert head.proj.out_features == WORD_VOCAB_SIZE


def test_forward_is_differentiable() -> None:
    # Arrange
    batch, t_enc, u_word = 2, 50, 5
    enc_features = torch.randn(batch, t_enc, D_ENC, requires_grad=True)
    word_targets = torch.randint(1, WORD_VOCAB_SIZE, (batch, u_word), dtype=torch.int64)
    input_lengths = torch.full((batch,), t_enc, dtype=torch.int64)
    target_lengths = torch.full((batch,), u_word, dtype=torch.int64)
    head = WordAuxCTCHead(d_enc=D_ENC, word_vocab_size=WORD_VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(enc_features, input_lengths, word_targets, target_lengths)
    loss.backward()

    # Assert
    assert enc_features.grad is not None
    assert torch.isfinite(enc_features.grad).all().item()


def test_forward_handles_short_input_without_nan() -> None:
    # Arrange
    batch, short_t, long_u = 2, 3, 6
    enc_features = torch.randn(batch, short_t, D_ENC)
    word_targets = torch.randint(1, WORD_VOCAB_SIZE, (batch, long_u), dtype=torch.int64)
    input_lengths = torch.full((batch,), short_t, dtype=torch.int64)
    target_lengths = torch.full((batch,), long_u, dtype=torch.int64)
    head = WordAuxCTCHead(d_enc=D_ENC, word_vocab_size=WORD_VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(enc_features, input_lengths, word_targets, target_lengths)

    # Assert
    assert torch.isfinite(loss).item()
