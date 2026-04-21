"""Tests for gp1.train.schedulers — build_noam and build_cosine_warmup.

TDD: RED phase.

Noam schedule: Vaswani et al. "Attention is All You Need" §5.3
  lr = d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)
  Peak at step == warmup_steps: d_model**-0.5 * warmup_steps**-0.5

Cosine warmup: linear ramp [0..lr] over warmup_steps, then cosine decay
  to lr * min_lr_ratio.

CONTRACTS.md §8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from gp1.train.schedulers import build_cosine_warmup, build_noam


def _dummy_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    """Tiny optimizer wrapping a single dummy parameter."""
    param = nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


# ---------------------------------------------------------------------------
# build_noam
# ---------------------------------------------------------------------------


def test_noam_lr_at_warmup_step_equals_peak():
    # Arrange
    d_model = 256
    warmup_steps = 4000
    peak = d_model**-0.5 * warmup_steps**-0.5  # theoretical peak

    opt = _dummy_optimizer(lr=1.0)
    sched = build_noam(opt, d_model=d_model, warmup_steps=warmup_steps)

    # Advance to warmup_steps
    for _ in range(warmup_steps):
        opt.step()
        sched.step()

    # Act
    current_lr = sched.get_last_lr()[0]

    # Assert: within floating point tolerance
    assert abs(current_lr - peak) < peak * 1e-6, (
        f"Noam peak mismatch: got {current_lr:.8f}, expected {peak:.8f}"
    )


def test_noam_lr_before_warmup_is_linear_in_step():
    # Arrange: during warmup, lr grows linearly.
    # lr(step) = d_model**-0.5 * step * warmup**-1.5
    # So lr(step2) / lr(step1) == step2 / step1 exactly.
    d_model = 64
    warmup_steps = 1000
    opt = _dummy_optimizer(lr=1.0)
    sched = build_noam(opt, d_model=d_model, warmup_steps=warmup_steps)

    def advance_to(target_step: int) -> float:
        for _ in range(target_step):
            opt.step()
            sched.step()
        return sched.get_last_lr()[0]

    # Capture lr at step 100 and step 200 (both < warmup)
    opt1 = _dummy_optimizer(lr=1.0)
    s1 = build_noam(opt1, d_model=d_model, warmup_steps=warmup_steps)
    opt2 = _dummy_optimizer(lr=1.0)
    s2 = build_noam(opt2, d_model=d_model, warmup_steps=warmup_steps)

    for _ in range(100):
        opt1.step()
        s1.step()
    lr_100 = s1.get_last_lr()[0]

    for _ in range(200):
        opt2.step()
        s2.step()
    lr_200 = s2.get_last_lr()[0]

    # Act / Assert: ratio should be 2.0 (linear)
    assert abs(lr_200 / lr_100 - 2.0) < 1e-6, (
        f"Noam warmup not linear: lr(100)={lr_100:.8f}, lr(200)={lr_200:.8f}"
    )


def test_noam_lr_after_warmup_decays_as_inverse_square_root():
    # Arrange: after warmup, lr(step) ∝ step**-0.5 → lr(2t)/lr(t) = 1/√2
    d_model = 64
    warmup_steps = 10
    opt1 = _dummy_optimizer(lr=1.0)
    s1 = build_noam(opt1, d_model=d_model, warmup_steps=warmup_steps)
    opt2 = _dummy_optimizer(lr=1.0)
    s2 = build_noam(opt2, d_model=d_model, warmup_steps=warmup_steps)

    target_t = 100  # well past warmup

    for _ in range(target_t):
        opt1.step()
        s1.step()
    lr_t = s1.get_last_lr()[0]

    for _ in range(2 * target_t):
        opt2.step()
        s2.step()
    lr_2t = s2.get_last_lr()[0]

    # Act / Assert
    expected_ratio = 1.0 / math.sqrt(2)
    actual_ratio = lr_2t / lr_t
    assert abs(actual_ratio - expected_ratio) < 1e-6, (
        f"Noam post-warmup decay wrong: lr({target_t})={lr_t:.8f}, "
        f"lr({2 * target_t})={lr_2t:.8f}, ratio={actual_ratio:.6f} (expected {expected_ratio:.6f})"
    )


# ---------------------------------------------------------------------------
# build_cosine_warmup
# ---------------------------------------------------------------------------


def test_cosine_warmup_lr_at_step_zero_is_near_zero():
    # Arrange: at step 0, warmup has just started → lr ≈ 0
    # LambdaLR is called AFTER the first step(), so we check the initial
    # get_last_lr() right after construction (step 0 = lambda(0)).
    total_steps = 1000
    warmup_steps = 100
    peak_lr = 0.01
    min_lr_ratio = 0.01

    opt = _dummy_optimizer(lr=peak_lr)
    sched = build_cosine_warmup(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # At construction, LambdaLR has not stepped yet → lr = base_lr * lambda(0)
    # lambda(0) = 0 / warmup_steps = 0.0
    initial_lr = opt.param_groups[0]["lr"]

    # Act: one step forward
    opt.step()
    sched.step()
    lr_step1 = sched.get_last_lr()[0]

    # Assert: lr at step 1 / warmup is roughly 1/warmup_steps of peak
    expected_step1 = peak_lr * (1.0 / warmup_steps)
    assert abs(lr_step1 - expected_step1) < peak_lr * 1e-6, (
        f"Cosine warmup at step 1: got {lr_step1:.8f}, expected {expected_step1:.8f}"
    )


def test_cosine_warmup_lr_at_warmup_end_equals_peak():
    # Arrange
    total_steps = 1000
    warmup_steps = 100
    peak_lr = 0.01
    min_lr_ratio = 0.01

    opt = _dummy_optimizer(lr=peak_lr)
    sched = build_cosine_warmup(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    for _ in range(warmup_steps):
        opt.step()
        sched.step()

    # Act
    lr_at_warmup_end = sched.get_last_lr()[0]

    # Assert
    assert abs(lr_at_warmup_end - peak_lr) < peak_lr * 1e-6, (
        f"Cosine warmup end mismatch: got {lr_at_warmup_end:.8f}, expected {peak_lr:.8f}"
    )


def test_cosine_warmup_lr_at_total_steps_equals_min_lr():
    # Arrange
    total_steps = 100
    warmup_steps = 10
    peak_lr = 0.01
    min_lr_ratio = 0.01
    min_lr = peak_lr * min_lr_ratio

    opt = _dummy_optimizer(lr=peak_lr)
    sched = build_cosine_warmup(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    for _ in range(total_steps):
        opt.step()
        sched.step()

    # Act
    lr_at_end = sched.get_last_lr()[0]

    # Assert
    assert abs(lr_at_end - min_lr) < peak_lr * 1e-5, (
        f"Cosine warmup end lr: got {lr_at_end:.8f}, expected {min_lr:.8f}"
    )


def test_cosine_warmup_lr_decreases_monotonically_after_warmup():
    # Arrange
    total_steps = 200
    warmup_steps = 20
    peak_lr = 0.01

    opt = _dummy_optimizer(lr=peak_lr)
    sched = build_cosine_warmup(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # Advance to end of warmup
    for _ in range(warmup_steps):
        opt.step()
        sched.step()

    # Act: collect LRs in decay phase
    lrs = []
    for _ in range(total_steps - warmup_steps):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # Assert: non-increasing after warmup
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-12, (
            f"LR increased at decay step {i}: {lrs[i]:.8f} -> {lrs[i + 1]:.8f}"
        )


def test_build_noam_returns_lr_scheduler_instance():
    # Arrange
    opt = _dummy_optimizer()

    # Act
    sched = build_noam(opt, d_model=128, warmup_steps=100)

    # Assert
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)


def test_build_cosine_warmup_returns_lr_scheduler_instance():
    # Arrange
    opt = _dummy_optimizer()

    # Act
    sched = build_cosine_warmup(opt, total_steps=200, warmup_steps=20)

    # Assert
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)
