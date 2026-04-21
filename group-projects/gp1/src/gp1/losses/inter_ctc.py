from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gp1.losses.ctc import CTCLoss


class InterCTCHead(nn.Module):
    def __init__(self, d_mid: int, vocab_size: int, blank_id: int = 0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_mid, vocab_size)
        self._ctc = CTCLoss(blank_id=blank_id)

    def forward(
        self,
        mid_features: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert mid_features.dim() == 3, "mid_features must be [B, T, D]"
        logits = self.proj(mid_features)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return self._ctc(log_probs, targets, input_lengths, target_lengths)
