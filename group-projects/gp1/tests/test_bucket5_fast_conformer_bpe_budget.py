"""Bucket-5 RED tests: FastConformerBPE param-budget guard (M2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gp1.models.fast_conformer_bpe import FastConformerBPE  # noqa: E402

_PARAM_BUDGET = 5_000_000


def test_fast_conformer_bpe_default_under_budget():
    """Default constructor must produce a model within the 5M param budget."""
    model = FastConformerBPE(vocab_size=35)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < _PARAM_BUDGET, (
        f"Default FastConformerBPE has {n_params:,} params — exceeds {_PARAM_BUDGET:,} budget"
    )


def test_fast_conformer_bpe_raises_on_oversized_config():
    """Constructing with d_model=512, n_blocks=30 must raise ValueError
    mentioning 'exceeds 5M'."""
    with pytest.raises(ValueError, match="exceeds 5M"):
        FastConformerBPE(vocab_size=35, d_model=512, n_blocks=30)
