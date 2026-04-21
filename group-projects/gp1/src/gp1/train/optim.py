"""Optimizers for GP1 ASR training.

CONTRACTS.md §8:
  build_novograd(params, lr, betas=(0.95, 0.5), weight_decay=1e-3) -> Optimizer
  build_adamw(params, lr, weight_decay=1e-6) -> Optimizer

NovoGrad is hand-rolled per Ginsburg et al. 2019 (arxiv:1905.11286) because
it is not available in ``torch.optim`` and the ``torch-optimizer`` package
is not installed in the project venv.

Key NovoGrad properties (from §3 of the paper):
  1. Gradient normalisation by the exponential moving average of the L2
     norm of the gradient (second moment in the L2 sense, NOT element-wise).
  2. Decoupled weight decay applied directly to the weights (same as AdamW),
     NOT folded into the gradient.
  3. Adam-style first-moment accumulation on the *normalised* gradient.

References:
  - Ginsburg et al. 2019 "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    https://arxiv.org/abs/1905.11286
  - NeMo reference implementation:
    https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/novograd.py
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NovoGrad implementation
# ---------------------------------------------------------------------------


class NovoGrad(Optimizer):
    """Stochastic Gradient Descent with Adaptive Gradient Norms (NovoGrad).

    Per-layer second moment is computed as the L2 norm of the gradient tensor
    (a single scalar per parameter group), NOT element-wise as in Adam.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        betas: Coefficients for first-moment (beta1) and second-moment (beta2)
            EMA. Defaults: (0.95, 0.5) as in Ginsburg et al. §4.
        eps: Small constant for numerical stability.
        weight_decay: Decoupled weight decay coefficient (AdamW-style).

    Reference:
        Ginsburg et al. 2019. https://arxiv.org/abs/1905.11286
        NeMo canonical impl: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/novograd.py
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.5),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # Lazy initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    # Second moment: L2 norm^2 of the first gradient batch
                    state["exp_avg_sq"] = torch.tensor(0.0, device=p.device)

                state["step"] += 1
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # --- Second moment update (L2 norm of gradient) ---
                grad_norm_sq = (grad * grad).sum()
                if state["step"] == 1:
                    exp_avg_sq.copy_(grad_norm_sq)
                else:
                    exp_avg_sq.mul_(beta2).add_(grad_norm_sq, alpha=1.0 - beta2)

                # --- Gradient normalisation ---
                denom = exp_avg_sq.sqrt().add_(eps)
                grad_normalised = grad / denom

                # --- First moment update on normalised gradient ---
                exp_avg.mul_(beta1).add_(grad_normalised, alpha=1.0 - beta1)

                # --- Decoupled weight decay (AdamW-style) ---
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # --- Parameter update ---
                p.add_(exp_avg, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Builder functions (CONTRACTS.md §8)
# ---------------------------------------------------------------------------


def build_novograd(
    params,
    lr: float,
    betas: tuple[float, float] = (0.95, 0.5),
    weight_decay: float = 1e-3,
) -> Optimizer:
    """Create a NovoGrad optimizer.

    Args:
        params: Model parameters or parameter groups.
        lr: Peak learning rate.
        betas: (beta1, beta2) — first and second moment decay coefficients.
            Defaults per Ginsburg et al. 2019 §4: (0.95, 0.5).
        weight_decay: Decoupled weight decay (default 1e-3).

    Returns:
        Configured NovoGrad optimizer instance.

    Reference:
        Ginsburg et al. 2019. https://arxiv.org/abs/1905.11286
    """
    return NovoGrad(params, lr=lr, betas=betas, weight_decay=weight_decay)


def build_adamw(
    params,
    lr: float,
    weight_decay: float = 1e-6,
) -> Optimizer:
    """Create an AdamW optimizer with GP1 defaults.

    Args:
        params: Model parameters or parameter groups.
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay (default 1e-6).

    Returns:
        Configured AdamW optimizer instance.
    """
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
