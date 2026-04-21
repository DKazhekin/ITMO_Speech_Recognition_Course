"""Tests for InterCTCHead — intermediate auxiliary CTC loss.

Reference: Lee & Watanabe 2021 "Intermediate Loss Regularization for CTC-based
Speech Recognition" (arXiv:2102.03216). NeMo's AuxiliaryInterCTCLoss in
nemo/collections/asr/losses/ctc.py implements the same pattern.
"""
from __future__ import annotations

import pytest
import torch

from gp1.losses.inter_ctc import InterCTCHead


VOCAB_SIZE = 35
BLANK_ID = 0
D_MID = 192


def test_forward_returns_finite_positive_scalar() -> None:
    # Arrange
    batch, t_mid, units = 2, 40, 10
    mid_features = torch.randn(batch, t_mid, D_MID)
    targets = torch.randint(1, VOCAB_SIZE, (batch, units), dtype=torch.int64)
    input_lengths = torch.full((batch,), t_mid, dtype=torch.int64)
    target_lengths = torch.full((batch,), units, dtype=torch.int64)
    head = InterCTCHead(d_mid=D_MID, vocab_size=VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(mid_features, input_lengths, targets, target_lengths)

    # Assert
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0


def test_projection_maps_mid_dim_to_vocab_size() -> None:
    # Arrange
    head = InterCTCHead(d_mid=D_MID, vocab_size=VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    out_features = head.proj.out_features
    in_features = head.proj.in_features

    # Assert
    assert in_features == D_MID
    assert out_features == VOCAB_SIZE


def test_forward_is_differentiable_wrt_mid_features() -> None:
    # Arrange
    batch, t_mid, units = 2, 40, 10
    mid_features = torch.randn(batch, t_mid, D_MID, requires_grad=True)
    targets = torch.randint(1, VOCAB_SIZE, (batch, units), dtype=torch.int64)
    input_lengths = torch.full((batch,), t_mid, dtype=torch.int64)
    target_lengths = torch.full((batch,), units, dtype=torch.int64)
    head = InterCTCHead(d_mid=D_MID, vocab_size=VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(mid_features, input_lengths, targets, target_lengths)
    loss.backward()

    # Assert
    assert mid_features.grad is not None
    assert torch.isfinite(mid_features.grad).all().item()


def test_forward_handles_short_input_without_nan() -> None:
    # Arrange: shorter than targets to force zero_infinity path
    batch, short_t, long_u = 2, 5, 12
    mid_features = torch.randn(batch, short_t, D_MID)
    targets = torch.randint(1, VOCAB_SIZE, (batch, long_u), dtype=torch.int64)
    input_lengths = torch.full((batch,), short_t, dtype=torch.int64)
    target_lengths = torch.full((batch,), long_u, dtype=torch.int64)
    head = InterCTCHead(d_mid=D_MID, vocab_size=VOCAB_SIZE, blank_id=BLANK_ID)

    # Act
    loss = head(mid_features, input_lengths, targets, target_lengths)

    # Assert
    assert torch.isfinite(loss).item()
