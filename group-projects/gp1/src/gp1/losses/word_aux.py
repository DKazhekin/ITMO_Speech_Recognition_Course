from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gp1.losses.ctc import CTCLoss


class WordAuxCTCHead(nn.Module):
    def __init__(self, d_enc: int, word_vocab_size: int, blank_id: int = 0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_enc, word_vocab_size)
        self._ctc = CTCLoss(blank_id=blank_id)

    def forward(
        self,
        enc_features: torch.Tensor,
        input_lengths: torch.Tensor,
        word_targets: torch.Tensor,
        word_target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert enc_features.dim() == 3, "enc_features must be [B, T, D]"
        logits = self.proj(enc_features)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return self._ctc(
            log_probs,
            input_lengths=input_lengths,
            targets=word_targets,
            target_lengths=word_target_lengths,
        )
