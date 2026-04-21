from __future__ import annotations

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank_id: int = 0) -> None:
        super().__init__()
        self.blank_id = blank_id
        self._ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert log_probs.dim() == 3, "log_probs must be [B, T, V]"
        # Cast to fp32 regardless of caller's dtype — CTC is numerically unstable in fp16.
        lp = log_probs.float().transpose(0, 1)
        with torch.autocast(device_type=lp.device.type, enabled=False):
            loss = self._ctc(lp, targets, input_lengths, target_lengths)
        return loss
