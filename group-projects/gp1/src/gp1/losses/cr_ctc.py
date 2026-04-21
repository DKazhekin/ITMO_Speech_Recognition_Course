from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRCTCLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, min_prob: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_prob = min_prob

    def forward(
        self,
        log_probs_a: torch.Tensor,
        log_probs_b: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert log_probs_a.dim() == 3, "log_probs must be [B, T, V]"
        batch, time, _ = log_probs_a.shape

        if self.temperature != 1.0:
            log_probs_a = F.log_softmax(log_probs_a / self.temperature, dim=-1)
            log_probs_b = F.log_softmax(log_probs_b / self.temperature, dim=-1)

        probs_a = log_probs_a.exp()
        probs_b = log_probs_b.exp()

        lengths_mask = (
            torch.arange(time, device=log_probs_a.device)
            .unsqueeze(0)
            .lt(input_lengths.unsqueeze(1))
        )

        if self.min_prob > 0.0:
            conf_a = probs_a.max(dim=-1).values >= self.min_prob
            conf_b = probs_b.max(dim=-1).values >= self.min_prob
            conf_mask = conf_a | conf_b
        else:
            conf_mask = torch.ones(
                batch, time, dtype=torch.bool, device=log_probs_a.device
            )

        mask = lengths_mask & conf_mask
        n_valid = mask.sum().clamp(min=1)

        # Symmetric KL: 0.5 * (KL(a||b) + KL(b||a))
        kl_ab = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)
        kl_ba = (probs_b * (log_probs_b - log_probs_a)).sum(dim=-1)
        sym_kl = 0.5 * (kl_ab + kl_ba)

        loss = (sym_kl * mask.float()).sum() / n_valid
        return loss
