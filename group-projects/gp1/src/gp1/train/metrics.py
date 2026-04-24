"""Character Error Rate (CER) metrics for GP1 ASR evaluation.

CONTRACTS.md §8:
  compute_cer(references, hypotheses) -> float
  compute_per_speaker_cer(references, hypotheses, spk_ids) -> dict[str, float]

CER is computed at the **corpus level**: sum of all edit distances divided
by sum of all reference lengths. This is NOT the mean of per-sample CERs;
longer references receive proportionally more weight.

Levenshtein DP is hand-rolled (30 lines) so the module has no runtime
dependency on `jiwer`. If `jiwer` becomes available in the project venv,
consider switching to `jiwer.cer()` — it uses the same corpus-level formula.

References:
  - Levenshtein, V.I. (1966). "Binary codes capable of correcting deletions,
    insertions, and reversals." Soviet Physics Doklady, 10(8), 707-710.
  - jiwer library (corpus-level CER): https://github.com/jitsi/jiwer
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: Levenshtein distance DP
# ---------------------------------------------------------------------------


def _edit_distance(ref: str, hyp: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses a space-optimised two-row DP table: O(min(|ref|, |hyp|)) space.

    Args:
        ref: Reference string (characters).
        hyp: Hypothesis string (characters).

    Returns:
        Integer edit distance (minimum insertions + deletions + substitutions).
    """
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return m
    if m == 0:
        return n

    # Ensure ref is the shorter one for space efficiency
    if n > m:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for j in range(1, m + 1):
        curr[0] = j
        for i in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[i] = prev[i - 1]
            else:
                curr[i] = 1 + min(prev[i - 1], prev[i], curr[i - 1])
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Compute corpus-level Character Error Rate (CER).

    CER = sum(edit_distance(ref_i, hyp_i)) / sum(len(ref_i))

    Empty reference strings contribute 0 to both numerator and denominator
    (they are silently skipped). If all references are empty, returns 0.0.

    Args:
        references: Ground-truth transcription strings.
        hypotheses: Model prediction strings (same order as references).

    Returns:
        CER as a float in [0.0, ∞). Values > 1.0 occur when the hypothesis
        is longer than the reference (net insertions).

    Raises:
        ValueError: If references and hypotheses have different lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"mismatched list lengths: {len(references)} references vs "
            f"{len(hypotheses)} hypotheses"
        )

    total_errors: int = 0
    total_ref_len: int = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len = len(ref)
        if ref_len == 0 and len(hyp) == 0:
            # Both empty: 0 errors, skip from denominator
            continue
        if ref_len == 0:
            # Empty ref with non-empty hyp: skip from denominator
            # (insertions relative to an empty reference are not penalised
            # in corpus-level CER — the reference has no characters to anchor)
            continue

        total_errors += _edit_distance(ref, hyp)
        total_ref_len += ref_len

    if total_ref_len == 0:
        return 0.0

    return total_errors / total_ref_len


def compute_per_speaker_cer(
    references: list[str],
    hypotheses: list[str],
    spk_ids: list[str],
) -> dict[str, float]:
    """Compute corpus-level CER independently for each speaker.

    Groups (ref, hyp) pairs by speaker ID and calls compute_cer within
    each group, preserving the corpus-level weighting by reference length.

    Args:
        references: Ground-truth transcription strings.
        hypotheses: Model prediction strings.
        spk_ids: Speaker identifier for each pair (same length as references).

    Returns:
        Mapping from speaker ID to that speaker's CER.

    Raises:
        ValueError: If any of the three input lists have different lengths.
    """
    if not (len(references) == len(hypotheses) == len(spk_ids)):
        raise ValueError(
            f"mismatched list lengths: references={len(references)}, "
            f"hypotheses={len(hypotheses)}, spk_ids={len(spk_ids)}"
        )

    # Group by speaker
    spk_refs: dict[str, list[str]] = {}
    spk_hyps: dict[str, list[str]] = {}

    for ref, hyp, spk in zip(references, hypotheses, spk_ids):
        if spk not in spk_refs:
            spk_refs[spk] = []
            spk_hyps[spk] = []
        spk_refs[spk].append(ref)
        spk_hyps[spk].append(hyp)

    return {spk: compute_cer(spk_refs[spk], spk_hyps[spk]) for spk in spk_refs}


def compute_digit_cer_in_out_harmonic(
    refs_digits: list[str],
    hyps_digits: list[str],
    spk_ids: list[str],
    in_domain_speakers: set[str],
) -> tuple[float, float, float]:
    """Split pairs by in/out-of-domain speaker and return (in_cer, out_cer, harmonic).

    Edge cases:
      * both subgroups empty -> (0.0, 0.0, 0.0)
      * one subgroup empty   -> harmonic = max(in_cer, out_cer) + warning
      * both CERs == 0       -> harmonic = 0.0 (avoids 0/0)

    Raises:
        ValueError: If input list lengths are not equal.
    """
    if not (len(refs_digits) == len(hyps_digits) == len(spk_ids)):
        raise ValueError(
            f"mismatched list lengths: refs_digits={len(refs_digits)}, "
            f"hyps_digits={len(hyps_digits)}, spk_ids={len(spk_ids)}"
        )

    in_refs: list[str] = []
    in_hyps: list[str] = []
    out_refs: list[str] = []
    out_hyps: list[str] = []

    for ref, hyp, spk in zip(refs_digits, hyps_digits, spk_ids):
        if spk in in_domain_speakers:
            in_refs.append(ref)
            in_hyps.append(hyp)
        else:
            out_refs.append(ref)
            out_hyps.append(hyp)

    if not in_refs and not out_refs:
        return 0.0, 0.0, 0.0

    in_cer = compute_cer(in_refs, in_hyps) if in_refs else 0.0
    out_cer = compute_cer(out_refs, out_hyps) if out_refs else 0.0

    if not in_refs or not out_refs:
        missing = "in-domain" if not in_refs else "out-of-domain"
        logger.warning(
            "harmonic_in_out_digit_cer: %s subgroup empty; falling back to max(in, out)",
            missing,
        )
        return in_cer, out_cer, max(in_cer, out_cer)

    if in_cer == 0.0 and out_cer == 0.0:
        return 0.0, 0.0, 0.0

    harmonic = 2.0 * in_cer * out_cer / (in_cer + out_cer)
    return in_cer, out_cer, harmonic
