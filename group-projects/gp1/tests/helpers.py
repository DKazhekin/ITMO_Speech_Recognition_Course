"""Shared test helpers for GP1 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_wav(path: Path, samplerate: int, duration_s: float = 0.5) -> None:
    """Write a silent WAV file at the given sample rate."""
    n_samples = int(samplerate * duration_s)
    data = np.zeros(n_samples, dtype=np.float32)
    sf.write(str(path), data, samplerate)
