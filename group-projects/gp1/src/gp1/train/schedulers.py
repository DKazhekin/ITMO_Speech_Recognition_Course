"""Learning rate schedulers for GP1 ASR training.

CONTRACTS.md §8:
  build_noam(optimizer, d_model, warmup_steps) -> LRScheduler
  build_cosine_warmup(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01)
      -> LRScheduler

Both are implemented via ``torch.optim.lr_scheduler.LambdaLR`` so they
compose with any ``torch.optim.Optimizer`` and are compatible with
``torch.optim.lr_scheduler.ChainedScheduler``.

LambdaLR indexing note:
  LambdaLR calls the lambda function with ``last_epoch`` as argument.
  At construction ``last_epoch=0`` (lambda(0) sets the initial lr).
  After each ``sched.step()``, ``last_epoch`` is incremented first, then
  lambda(last_epoch) is called to set the new lr.
  So after N calls to sched.step(), get_last_lr() reflects lambda(N).
  We interpret step N as "we have completed N gradient steps" — therefore
  step 0 means "no gradient steps yet" and is the initial LR.

References:
  Noam schedule:
    Vaswani et al. 2017 "Attention Is All You Need" §5.3
    https://arxiv.org/abs/1706.03762

  Cosine schedule with warmup:
    Loshchilov & Hutter 2016 "SGDR: Stochastic Gradient Descent with
    Warm Restarts" https://arxiv.org/abs/1608.03983
"""

from __future__ import annotations

import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noam schedule (Vaswani et al. 2017 §5.3)
# ---------------------------------------------------------------------------


def build_noam(
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int,
) -> LRScheduler:
    """Create a Noam (transformer) learning rate scheduler.

    The schedule (Vaswani et al. 2017 §5.3):
        lr(step) = base_lr * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)

    Peak at step == warmup_steps:
        peak = base_lr * d_model**-0.5 * warmup_steps**-0.5

    LambdaLR indexing: after N calls to sched.step(), get_last_lr() returns
    base_lr * lambda(N). So lambda(N) encodes the schedule at step N.

    Args:
        optimizer: Optimizer to attach the schedule to.
        d_model: Model dimension (normalisation constant from the paper).
        warmup_steps: Number of linear warm-up gradient steps.

    Returns:
        LambdaLR scheduler instance.

    Reference:
        Vaswani et al. 2017. https://arxiv.org/abs/1706.03762  §5.3
    """
    scale = d_model**-0.5

    def _noam_lambda(step: int) -> float:
        # step == last_epoch; step 0 → initial lr (before any gradient steps).
        # Clamp to 1 to avoid division by zero; step 0 uses warmup branch.
        s = max(step, 1)
        return scale * min(s**-0.5, s * (warmup_steps**-1.5))

    return LambdaLR(optimizer, lr_lambda=_noam_lambda)


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------


def build_cosine_warmup(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> LRScheduler:
    """Create a cosine-decay schedule with a linear warmup phase.

    Phase 1 — linear warmup over ``warmup_steps`` gradient steps:
        lr(step) = base_lr * step / warmup_steps
        so lr(0) = 0, lr(warmup_steps) = base_lr.

    Phase 2 — cosine decay from ``base_lr`` to ``base_lr * min_lr_ratio``:
        lr(step) = base_lr * (min_lr_ratio + 0.5*(1-min_lr_ratio)
                   * (1 + cos(π * progress)))
    where progress ∈ [0, 1] runs from warmup_steps to total_steps.

    Args:
        optimizer: Optimizer to attach the schedule to.
        total_steps: Total gradient steps (end of cosine decay).
        warmup_steps: Number of linear warm-up gradient steps.
        min_lr_ratio: Minimum LR as a fraction of peak. Default 0.01.

    Returns:
        LambdaLR scheduler instance.

    Reference:
        Loshchilov & Hutter 2016. https://arxiv.org/abs/1608.03983
    """
    decay_steps = total_steps - warmup_steps

    def _cosine_lambda(step: int) -> float:
        # step == last_epoch (0-based; matches gradient step count).
        if step <= warmup_steps:
            # Linear ramp: step/warmup_steps
            # step 0 → 0.0, step warmup_steps → 1.0
            return step / max(warmup_steps, 1)
        # Cosine decay phase
        progress = (step - warmup_steps) / max(decay_steps, 1)
        progress = min(progress, 1.0)
        cos_val = math.cos(math.pi * progress)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + cos_val)

    return LambdaLR(optimizer, lr_lambda=_cosine_lambda)
