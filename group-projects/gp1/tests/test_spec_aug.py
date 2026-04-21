"""Tests for SpecAugmenter (CONTRACTS.md §4)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_no_masking_applied_in_eval_mode() -> None:
    # Arrange
    from gp1.data.spec_aug import SpecAugmenter

    torch.manual_seed(0)
    mel = torch.randn(2, 80, 200)
    lengths = torch.tensor([200, 150], dtype=torch.int64)

    aug = SpecAugmenter()
    aug.eval()

    # Act
    out = aug(mel.clone(), lengths)

    # Assert
    torch.testing.assert_close(out, mel)


def test_masking_applied_in_train_mode_changes_input() -> None:
    # Arrange
    from gp1.data.spec_aug import SpecAugmenter

    torch.manual_seed(1)
    mel = torch.randn(2, 80, 200)
    lengths = torch.tensor([200, 200], dtype=torch.int64)

    aug = SpecAugmenter()
    aug.train()

    # Act
    out = aug(mel.clone(), lengths)

    # Assert — something was masked (differs from input)
    assert out.shape == mel.shape
    assert not torch.allclose(out, mel)


def test_shape_is_preserved() -> None:
    # Arrange
    from gp1.data.spec_aug import SpecAugmenter

    torch.manual_seed(2)
    mel = torch.randn(3, 80, 137)
    lengths = torch.tensor([137, 100, 80], dtype=torch.int64)

    aug = SpecAugmenter()
    aug.train()

    # Act
    out = aug(mel.clone(), lengths)

    # Assert
    assert out.shape == mel.shape


def test_contract_params_match_f_15x2_t_25x5() -> None:
    # Arrange + Act
    from gp1.data.spec_aug import SpecAugmenter

    aug = SpecAugmenter()

    # Assert — CONTRACTS.md §1 defaults
    assert aug.freq_mask_param == 15
    assert aug.num_freq_masks == 2
    assert aug.time_mask_param == 25
    assert aug.num_time_masks == 5
    # no time-warp is an explicit non-goal (see plan)
    assert not hasattr(aug, "time_warp_param") or aug.time_warp_param in (0, None)
