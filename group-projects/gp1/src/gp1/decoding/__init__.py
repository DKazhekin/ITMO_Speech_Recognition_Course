"""Decoding module for GP1 Russian ASR.

Public API:
  - greedy_decode: argmax CTC decoding (always available)
  - KenLMWrapper: KenLM language model wrapper (requires kenlm)
  - BeamSearchConfig: frozen dataclass for beam search hyperparameters
  - BeamSearchDecoder: pyctcdecode-backed beam search (requires pyctcdecode)
"""

from gp1.decoding.greedy import greedy_decode
from gp1.decoding.lm import KenLMWrapper
from gp1.decoding.beam_pyctc import BeamSearchConfig, BeamSearchDecoder

__all__ = [
    "greedy_decode",
    "KenLMWrapper",
    "BeamSearchConfig",
    "BeamSearchDecoder",
]
