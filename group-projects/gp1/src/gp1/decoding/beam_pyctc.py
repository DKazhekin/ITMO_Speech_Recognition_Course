"""Beam-search CTC decoding via pyctcdecode.

Wraps pyctcdecode.build_ctcdecoder with:
  - Hard lexicon (unigrams) constraining outputs to known number words.
  - Optional KenLM 4-gram language model for shallow fusion.
  - Hotword boosting for тысяча/тысячи/тысяч.

Label convention (pyctcdecode requirement):
  labels[0] = ""  — blank token
  labels[1] = " " — word separator / space
  labels[2..] = Russian lowercase letters in RUSSIAN_ALPHABET_LOWER order

References:
  - CONTRACTS.md §7: BeamSearchDecoder / BeamSearchConfig signatures
  - pyctcdecode: https://github.com/kensho-technologies/pyctcdecode
    build_ctcdecoder(labels, kenlm_model_path, unigrams, alpha, beta)
    BeamSearchDecoderCTC.decode_beams(logits, beam_width, hotwords, hotword_weight)
    Returns list of (text, last_lm_state, text_frames, logit_score, lm_score)
  - Plan (merry-hugging-flurry.md): alpha=0.7, beta=1.0, beam_width=100,
    hotword_weight=8.0 on тысяча/тысячи/тысяч
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gp1.text.vocab import CharVocab, RUSSIAN_ALPHABET_LOWER

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _labels_from_char_vocab(vocab: CharVocab) -> list[str]:
    """Build the pyctcdecode label list from a CharVocab.

    pyctcdecode expects:
      index 0  → "" (blank token — pyctcdecode uses "" not "<blank>")
      index 1  → " " (space / word separator)
      index 2+ → individual characters

    CharVocab assigns:
      id 0 = blank, id 1 = space, ids 2..34 = Russian letters

    Returns:
        List of length vocab.vocab_size where position i holds the
        character string for token id i.
    """
    labels: list[str] = [""] * vocab.vocab_size
    labels[0] = ""  # blank → empty string (pyctcdecode convention)
    labels[1] = " "  # space
    for idx, ch in enumerate(RUSSIAN_ALPHABET_LOWER):
        labels[idx + 2] = ch
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamSearchConfig:
    """Configuration for pyctcdecode beam search.

    Defaults from plan (merry-hugging-flurry.md §Decoding):
      alpha=0.7, beta=1.0, beam_width=100, hotword_weight=8.0
    """

    alpha: float = 0.7
    beta: float = 1.0
    beam_width: int = 100
    hotwords: tuple[str, ...] = ("тысяча", "тысячи", "тысяч")
    hotword_weight: float = 8.0


class BeamSearchDecoder:
    """CTC beam-search decoder backed by pyctcdecode.

    Args:
        vocab: CharVocab defining the token ↔ id mapping.
        kenlm_path: Path to compiled KenLM binary (``*.bin``), or None for
            pure hard-lexicon mode without a language model.
        unigrams: Hard lexicon — list of allowed word strings.  pyctcdecode
            uses this as a whitelist during beam expansion.
        config: Hyperparameters (alpha, beta, beam_width, hotwords,
            hotword_weight).
    """

    def __init__(
        self,
        vocab: CharVocab,
        kenlm_path: Path | None,
        unigrams: list[str],
        config: BeamSearchConfig,
    ) -> None:
        import pyctcdecode  # lazy import — not available in all envs

        self.config = config
        self.labels = _labels_from_char_vocab(vocab)

        kenlm_model_path: str | None = (
            str(kenlm_path) if kenlm_path is not None else None
        )

        self._decoder = pyctcdecode.build_ctcdecoder(
            labels=self.labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=config.alpha,
            beta=config.beta,
        )
        log.info(
            "BeamSearchDecoder: built decoder. kenlm=%s, unigrams=%d, "
            "alpha=%.2f, beta=%.2f, beam_width=%d",
            kenlm_model_path or "none",
            len(unigrams),
            config.alpha,
            config.beta,
            config.beam_width,
        )

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> list[str]:
        """Decode a batch of CTC log-probability tensors.

        Args:
            log_probs: [B, T, V] float tensor. Slices are truncated to
                output_lengths[i] before being passed to pyctcdecode.
            output_lengths: [B] int64 tensor of valid frame counts.

        Returns:
            List of B decoded strings (best beam hypothesis per sample).
        """
        assert log_probs.ndim == 3, (
            f"log_probs must be [B, T, V], got {tuple(log_probs.shape)}"
        )
        assert output_lengths.ndim == 1, (
            f"output_lengths must be [B], got {tuple(output_lengths.shape)}"
        )
        B = log_probs.size(0)
        assert output_lengths.size(0) == B

        results: list[str] = []
        hotwords = list(self.config.hotwords)
        beam_width = self.config.beam_width
        hotword_weight = self.config.hotword_weight

        for i in range(B):
            length = int(output_lengths[i].item())
            # Truncate to valid frames and move to CPU numpy
            logits_np: np.ndarray = log_probs[i, :length, :].cpu().float().numpy()

            beams = self._decoder.decode_beams(
                logits_np,
                beam_width=beam_width,
                hotwords=hotwords,
                hotword_weight=hotword_weight,
            )
            # beams: list of (text, last_lm_state, text_frames, logit_score, lm_score)
            # The first element is the best hypothesis.
            best_text: str = beams[0][0] if beams else ""
            results.append(best_text)

        log.debug("BeamSearchDecoder.decode_batch: B=%d decoded", B)
        return results
