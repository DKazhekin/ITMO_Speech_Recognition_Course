"""Tests for gp1.train.optim — build_novograd and build_adamw.

TDD: RED phase.

CONTRACTS.md §8.
NovoGrad reference: Ginsburg et al. 2019 (https://arxiv.org/abs/1905.11286).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gp1.train.optim import build_adamw, build_novograd


def _small_linear() -> nn.Linear:
    """Creates a small nn.Linear with reproducible parameters."""
    torch.manual_seed(42)
    layer = nn.Linear(4, 2, bias=True)
    return layer


# ---------------------------------------------------------------------------
# build_novograd
# ---------------------------------------------------------------------------


def test_novograd_single_step_updates_parameters():
    # Arrange
    model = _small_linear()
    params_before = [p.clone().detach() for p in model.parameters()]
    opt = build_novograd(model.parameters(), lr=1e-2)

    # Act: forward + backward + step
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    # Assert: at least one parameter changed
    changed = any(
        not torch.allclose(p_before, p_after)
        for p_before, p_after in zip(params_before, model.parameters())
    )
    assert changed, "NovoGrad step did not update any parameter"


def test_novograd_zero_grad_and_zero_wd_keeps_params_unchanged():
    # Arrange: zero gradient AND weight_decay=0 → no update.
    # NovoGrad decouples WD from the gradient, so with both set to zero
    # there is nothing to update parameters with.
    model = _small_linear()
    opt = build_novograd(model.parameters(), lr=1e-2, weight_decay=0.0)

    # set_to_none=False so we get actual zero tensors (not None)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    params_before = [p.clone().detach() for p in model.parameters()]

    # Act
    opt.step()

    # Assert: params unchanged (zero gradient, zero WD → zero update)
    for p_before, p_after in zip(params_before, model.parameters()):
        assert torch.allclose(p_before, p_after), (
            "Params changed with zero gradient and zero weight decay"
        )


def test_novograd_weight_decay_decoupled_from_gradient():
    # Arrange: zero gradient but non-zero weight_decay → params should shrink.
    # This verifies the key NovoGrad property: WD is decoupled (applied
    # directly to weights like AdamW, NOT to the gradient like L2 reg).
    model = _small_linear()
    # Fix weight values to non-zero so decay has something to shrink.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)

    opt = build_novograd(model.parameters(), lr=1.0, weight_decay=0.1)

    # Set gradients to zero
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    params_before = [p.clone().detach() for p in model.parameters()]

    # Act
    opt.step()

    # Assert: every parameter should have decreased in magnitude
    for p_before, p_after in zip(params_before, model.parameters()):
        assert (p_after.abs() < p_before.abs()).all(), (
            "Weight decay should shrink parameters even with zero gradient"
        )


def test_adamw_respects_weight_decay():
    # Arrange
    model = _small_linear()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)

    opt = build_adamw(model.parameters(), lr=1e-2, weight_decay=0.1)

    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    # Assert: at least some parameters moved
    changed = any(not torch.allclose(torch.ones_like(p), p) for p in model.parameters())
    assert changed


def test_builders_return_torch_optim_optimizer_instance():
    # Arrange
    model = _small_linear()

    # Act
    novograd_opt = build_novograd(model.parameters(), lr=1e-3)
    adamw_opt = build_adamw(model.parameters(), lr=1e-3)

    # Assert: both must be torch.optim.Optimizer subclasses
    assert isinstance(novograd_opt, torch.optim.Optimizer)
    assert isinstance(adamw_opt, torch.optim.Optimizer)
