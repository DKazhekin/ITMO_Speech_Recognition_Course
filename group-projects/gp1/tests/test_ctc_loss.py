"""Tests for CTCLoss wrapper.

Ensures fp32 stability, zero_infinity behavior, and no NaN under edge cases
including very short input (T<U).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from gp1.losses.ctc import CTCLoss


# Shape constants matching CharVocab.vocab_size = 35 (blank + space + 33 letters).
VOCAB_SIZE = 35
BLANK_ID = 0


def _random_log_probs(batch: int, time: int, vocab: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, time, vocab, generator=generator)
    return F.log_softmax(logits, dim=-1)


def test_forward_returns_finite_positive_scalar_on_normal_inputs():
    # Arrange
    batch, time, units = 2, 50, 10
    log_probs = _random_log_probs(batch, time, VOCAB_SIZE)
    targets = torch.randint(1, VOCAB_SIZE, (batch, units), dtype=torch.int64)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    target_lengths = torch.full((batch,), units, dtype=torch.int64)
    loss_fn = CTCLoss(blank_id=BLANK_ID)

    # Act
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Assert
    assert loss.dim() == 0, "loss must be a 0-dim scalar"
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0


def test_forward_no_nan_when_input_equals_target_length():
    # Arrange: T == U (no blanks to insert between symbols)
    batch, t_eq_u = 2, 10
    log_probs = _random_log_probs(batch, t_eq_u, VOCAB_SIZE)
    targets = torch.randint(1, VOCAB_SIZE, (batch, t_eq_u), dtype=torch.int64)
    input_lengths = torch.full((batch,), t_eq_u, dtype=torch.int64)
    target_lengths = torch.full((batch,), t_eq_u, dtype=torch.int64)
    loss_fn = CTCLoss(blank_id=BLANK_ID)

    # Act
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Assert
    assert torch.isfinite(loss).item(), "loss must be finite when T == U"


def test_forward_stays_finite_when_input_shorter_than_target():
    # Arrange: T < U; zero_infinity=True must prevent inf/nan
    batch, short_t, long_u = 2, 5, 12
    log_probs = _random_log_probs(batch, short_t, VOCAB_SIZE)
    targets = torch.randint(1, VOCAB_SIZE, (batch, long_u), dtype=torch.int64)
    input_lengths = torch.full((batch,), short_t, dtype=torch.int64)
    target_lengths = torch.full((batch,), long_u, dtype=torch.int64)
    loss_fn = CTCLoss(blank_id=BLANK_ID)

    # Act
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Assert
    assert torch.isfinite(loss).item(), "zero_infinity must clamp inf to finite"
    assert not math.isnan(loss.item())


def test_forward_remains_fp32_even_when_inputs_are_fp16():
    # Arrange: verify internal fp32 cast by feeding half-precision log_probs.
    batch, time, units = 2, 20, 5
    log_probs = _random_log_probs(batch, time, VOCAB_SIZE).to(torch.float16)
    targets = torch.randint(1, VOCAB_SIZE, (batch, units), dtype=torch.int64)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    target_lengths = torch.full((batch,), units, dtype=torch.int64)
    loss_fn = CTCLoss(blank_id=BLANK_ID)

    # Act
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Assert
    assert loss.dtype == torch.float32, "CTC must run in fp32 regardless of input"
    assert torch.isfinite(loss).item()


def test_forward_is_differentiable_wrt_log_probs():
    # Arrange
    batch, time, units = 2, 30, 8
    log_probs = _random_log_probs(batch, time, VOCAB_SIZE).requires_grad_(True)
    targets = torch.randint(1, VOCAB_SIZE, (batch, units), dtype=torch.int64)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    target_lengths = torch.full((batch,), units, dtype=torch.int64)
    loss_fn = CTCLoss(blank_id=BLANK_ID)

    # Act
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
    loss.backward()

    # Assert
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all().item()
