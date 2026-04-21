"""Training infrastructure for GP1 ASR system.

CONTRACTS.md §8: metrics, optimizers, schedulers, trainer.
"""

from __future__ import annotations

from gp1.train.metrics import compute_cer, compute_per_speaker_cer
from gp1.train.optim import build_adamw, build_novograd
from gp1.train.schedulers import build_cosine_warmup, build_noam
from gp1.train.trainer import Trainer, TrainerConfig

__all__ = [
    "compute_cer",
    "compute_per_speaker_cer",
    "build_novograd",
    "build_adamw",
    "build_noam",
    "build_cosine_warmup",
    "Trainer",
    "TrainerConfig",
]
