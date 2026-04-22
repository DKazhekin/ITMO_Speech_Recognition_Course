"""Bucket-5: T'/U_max CTC safety margin test parametrized over all 4 architectures.

Extends test_models_shapes.py to cover CRDNN, EfficientConformer, and
FastConformerBPE in addition to QuartzNet.

For each architecture, we run the worst-case sample (T=350 mel frames,
U_max = 59 chars for "999999" in Russian) and assert that T'/U_max
meets a per-model minimum ratio (documented and enforced by the
architecture's module docstring).

Ratios:
  - QuartzNet10x4 (s=2): T'/U_max = 175/59 >= 2.0
  - CRDNN         (s=2): T'/U_max = 175/59 >= 2.0   (s=1 gives 350/59 ≈5.9)
  - EfficientConformer (s=4): T'/U_max = 87/59 >= 1.4  (documented in module docstring)
  - FastConformerBPE (s=4, BPE U_max~30): T'/U_max = 87/30 >= 2.0  (BPE vocab shrinks U)

References:
  - EfficientConformer docstring: T'/U_max ≈ 1.47 for worst-case char vocab
  - gp1_subsample_and_kernel_constraints.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Worst-case for char vocab: "999999" in Russian = 59 chars
U_MAX_CHAR = 59
# BPE U_max is ~30 for "999999" in Russian words (fewer BPE tokens than chars)
U_MAX_BPE = 30
# 3.5s of fast speech at 16kHz / hop=160 -> 350 frames
T_WORST_CASE = 350
# Standard char vocab size (blank + space + 33 Russian)
CHAR_VOCAB_SIZE = 35
# Minimal BPE vocab size valid for instantiation
BPE_VOCAB_SIZE = 65


def _has_sentencepiece() -> bool:
    try:
        import sentencepiece  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Parametrized test: T'/U_max >= min_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_id,vocab_size,u_max,min_ratio",
    [
        # QuartzNet: s=2 → T'=175, ratio=175/59≈2.97
        ("quartznet_10x4", CHAR_VOCAB_SIZE, U_MAX_CHAR, 2.0),
        # CRDNN: default s=1 → T'=350, ratio=350/59≈5.9
        ("crdnn", CHAR_VOCAB_SIZE, U_MAX_CHAR, 2.0),
        # EfficientConformer: s=4 → T'≈87 (ceil-div twice), ratio≈1.47
        # Lower threshold matches the ACCEPTED value in the module docstring.
        ("efficient_conformer", CHAR_VOCAB_SIZE, U_MAX_CHAR, 1.4),
        # FastConformerBPE: s=4, BPE U_max≈30 → ratio=87/30≈2.9
        pytest.param(
            "fast_conformer_bpe",
            BPE_VOCAB_SIZE,
            U_MAX_BPE,
            2.0,
            marks=pytest.mark.skipif(
                not _has_sentencepiece(),
                reason="sentencepiece not installed — BPE model test skipped",
            ),
        ),
    ],
    ids=["quartznet_10x4", "crdnn", "efficient_conformer", "fast_conformer_bpe"],
)
def test_T_prime_over_U_max_geq_min_ratio(arch_id, vocab_size, u_max, min_ratio):
    """T'/U_max >= min_ratio on worst-case 3.5 s sample for all architectures."""
    model = _build_model(arch_id, vocab_size)
    model.eval()

    mel = torch.randn(1, 80, T_WORST_CASE)
    mel_lengths = torch.tensor([T_WORST_CASE], dtype=torch.long)
    with torch.no_grad():
        out = model(mel, mel_lengths)

    t_prime = int(out.output_lengths[0].item())
    ratio = t_prime / u_max

    assert ratio >= min_ratio, (
        f"{arch_id}: T'/U_max = {t_prime}/{u_max} = {ratio:.3f} < {min_ratio} "
        f"— CTC training will be unstable (s={model.subsample_factor})"
    )


# ---------------------------------------------------------------------------
# Helper: instantiate each model with default params
# ---------------------------------------------------------------------------


def _build_model(arch_id: str, vocab_size: int):
    if arch_id == "quartznet_10x4":
        from gp1.models.quartznet import QuartzNet10x4

        torch.manual_seed(0)
        return QuartzNet10x4(vocab_size=vocab_size)

    if arch_id == "crdnn":
        from gp1.models.crdnn import CRDNN

        torch.manual_seed(0)
        return CRDNN(vocab_size=vocab_size)

    if arch_id == "efficient_conformer":
        from gp1.models.efficient_conformer import EfficientConformer

        torch.manual_seed(0)
        return EfficientConformer(vocab_size=vocab_size)

    if arch_id == "fast_conformer_bpe":
        from gp1.models.fast_conformer_bpe import FastConformerBPE

        torch.manual_seed(0)
        return FastConformerBPE(vocab_size=vocab_size)

    raise ValueError(f"Unknown arch_id: {arch_id!r}")
