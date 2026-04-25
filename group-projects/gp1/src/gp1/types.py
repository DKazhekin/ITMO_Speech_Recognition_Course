"""Shared types for the GP1 ASR pipeline (CONTRACTS.md §1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ManifestRecord:
    audio_path: Path  # absolute path to .wav/.mp3
    transcription: str  # "139473" (digit string, 4..6 digits)
    spk_id: str  # "spk_A" ... "spk_N"
    gender: str  # "male" | "female"
    ext: str  # "wav" | "mp3"
    samplerate: int  # native sample rate (will be resampled to 16 kHz)
    duration_s: float = 0.0  # audio duration in seconds (frames / samplerate)


@dataclass
class Batch:
    audio: torch.Tensor  # [B, T_audio_max] float32, 16 kHz, zero-padded
    audio_lengths: torch.Tensor  # [B] int64, actual audio length in samples
    targets: torch.Tensor  # [B, U_max] int64, char-ids (or word-ids)
    target_lengths: torch.Tensor  # [B] int64
    spk_ids: list[str]  # [B]
    transcriptions: list[str]  # [B] digit strings, for metric computation


@dataclass(frozen=True)
class AugConfig:
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    speed_prob: float = 1.0
    pitch_prob: float = 0.3
    pitch_range_semitones: tuple[float, float] = (-3.0, 3.0)
    gain_prob: float = 0.7
    gain_db_range: tuple[float, float] = (-8.0, 8.0)
    seed: int | None = None
