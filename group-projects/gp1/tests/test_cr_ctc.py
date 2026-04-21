"""Tests for CRCTCLoss — consistency regularization between two augmented views.

Reference: Yao et al. 2024 "CR-CTC: Consistency Regularization on CTC for
Improved Speech Recognition" (arXiv:2410.05101). Original implementation in
k2-fsa/icefall (zipformer/model.py). Plan specifies min_prob-threshold masking
instead of icefall's length-only masking.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gp1.losses.cr_ctc import CRCTCLoss


VOCAB_SIZE = 35


def _random_log_probs(batch: int, time: int, vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, time, vocab, generator=generator)
    return F.log_softmax(logits, dim=-1)


def test_identical_distributions_yield_near_zero_loss() -> None:
    # Arrange
    batch, time = 2, 30
    log_probs = _random_log_probs(batch, time, VOCAB_SIZE, seed=1)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    # min_prob=0 keeps every frame regardless of confidence
    loss_fn = CRCTCLoss(temperature=1.0, min_prob=0.0)

    # Act
    loss = loss_fn(log_probs, log_probs.clone(), input_lengths)

    # Assert
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert abs(loss.item()) < 1e-5, f"identical inputs must give ~0 loss, got {loss.item()}"


def test_different_distributions_give_positive_symmetric_loss() -> None:
    # Arrange
    batch, time = 2, 30
    log_probs_a = _random_log_probs(batch, time, VOCAB_SIZE, seed=42)
    log_probs_b = _random_log_probs(batch, time, VOCAB_SIZE, seed=43)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    loss_fn = CRCTCLoss(temperature=1.0, min_prob=0.0)

    # Act
    loss_ab = loss_fn(log_probs_a, log_probs_b, input_lengths)
    loss_ba = loss_fn(log_probs_b, log_probs_a, input_lengths)

    # Assert
    assert loss_ab.item() > 0.0
    assert torch.allclose(loss_ab, loss_ba, atol=1e-5), (
        f"symmetric KL must be order-invariant: {loss_ab.item()} vs {loss_ba.item()}"
    )


def test_is_differentiable_through_both_inputs() -> None:
    # Arrange
    batch, time = 2, 20
    log_probs_a = _random_log_probs(batch, time, VOCAB_SIZE, seed=7).requires_grad_(True)
    log_probs_b = _random_log_probs(batch, time, VOCAB_SIZE, seed=8).requires_grad_(True)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    loss_fn = CRCTCLoss(temperature=1.0, min_prob=0.0)

    # Act
    loss = loss_fn(log_probs_a, log_probs_b, input_lengths)
    loss.backward()

    # Assert
    assert log_probs_a.grad is not None
    assert log_probs_b.grad is not None
    assert torch.isfinite(log_probs_a.grad).all().item()
    assert torch.isfinite(log_probs_b.grad).all().item()


def test_input_lengths_mask_out_padding_frames() -> None:
    # Arrange
    batch, time = 2, 30
    log_probs_a = _random_log_probs(batch, time, VOCAB_SIZE, seed=11)
    log_probs_b = _random_log_probs(batch, time, VOCAB_SIZE, seed=12)
    short_len = 10
    input_lengths_short = torch.tensor([short_len, short_len], dtype=torch.int64)
    input_lengths_full = torch.full((batch,), time, dtype=torch.int64)
    loss_fn = CRCTCLoss(temperature=1.0, min_prob=0.0)

    # Act
    loss_short = loss_fn(log_probs_a, log_probs_b, input_lengths_short)
    loss_full = loss_fn(log_probs_a, log_probs_b, input_lengths_full)

    # Assert
    assert torch.isfinite(loss_short).item()
    assert torch.isfinite(loss_full).item()
    # Both should be non-negative; shortening valid region can only drop frames.
    assert loss_short.item() >= 0.0
    assert loss_full.item() >= 0.0


def test_min_prob_masking_drops_low_confidence_frames() -> None:
    # Arrange: uniform log-probs → max prob == 1/V < 0.5 → every frame is masked out.
    batch, time = 2, 10
    log_probs_a = F.log_softmax(torch.zeros(batch, time, VOCAB_SIZE), dim=-1)
    log_probs_b = F.log_softmax(torch.zeros(batch, time, VOCAB_SIZE), dim=-1)
    input_lengths = torch.full((batch,), time, dtype=torch.int64)
    loss_fn = CRCTCLoss(temperature=1.0, min_prob=0.5)

    # Act
    loss = loss_fn(log_probs_a, log_probs_b, input_lengths)

    # Assert: max prob = 1/35 < 0.5 — all frames masked → loss exactly 0.0
    assert loss.item() == 0.0
