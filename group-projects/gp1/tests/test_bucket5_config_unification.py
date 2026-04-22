"""Bucket-5 RED tests: config unification (M1) and script wiring.

Tests for gp1.config.load_config and verifying that scripts no longer
embed their own config-loading logic.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
for _p in (_REPO_SRC, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gp1.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build minimal test configs in a tmp dir
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test 1: single-parent defaults resolution
# ---------------------------------------------------------------------------


def test_load_config_resolves_single_parent_defaults():
    """Child config with defaults:[base] — child keys override base keys."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_yaml(
            tmp_path / "base.yaml",
            {
                "audio": {"samplerate": 16000, "n_mels": 80},
                "train": {"max_epochs": 50, "grad_accum": 1},
            },
        )
        _write_yaml(
            tmp_path / "child.yaml",
            {
                "defaults": ["base"],
                "train": {"max_epochs": 100},
            },
        )
        result = load_config(tmp_path / "child.yaml")

    # Child's overridden value wins.
    assert result["train"]["max_epochs"] == 100
    # Base value for sibling key is preserved (shallow dict merge).
    assert result["train"]["grad_accum"] == 1
    # Key from base not mentioned in child is still present.
    assert result["audio"]["samplerate"] == 16000
    # 'defaults' key must NOT appear in the output.
    assert "defaults" not in result


# ---------------------------------------------------------------------------
# Test 2: multi-parent defaults resolution
# ---------------------------------------------------------------------------


def test_load_config_resolves_multi_parent_defaults():
    """defaults:[base, extras] — later parent wins over earlier; child wins over both."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_yaml(
            tmp_path / "base.yaml",
            {"audio": {"samplerate": 16000}, "train": {"lr": 1e-3, "epochs": 10}},
        )
        _write_yaml(
            tmp_path / "extras.yaml",
            {"train": {"epochs": 20, "extra_key": "hello"}},
        )
        _write_yaml(
            tmp_path / "child.yaml",
            {
                "defaults": ["base", "extras"],
                "train": {"epochs": 99},
            },
        )
        result = load_config(tmp_path / "child.yaml")

    # Child overrides all parents.
    assert result["train"]["epochs"] == 99
    # Key only in base — preserved.
    assert result["train"]["lr"] == 1e-3
    # Key only in extras — preserved.
    assert result["train"]["extra_key"] == "hello"
    # Key only in base top-level — preserved.
    assert result["audio"]["samplerate"] == 16000
    assert "defaults" not in result


# ---------------------------------------------------------------------------
# Test 3: child nested-dict override preserves sibling keys
# ---------------------------------------------------------------------------


def test_load_config_child_override_nested_dict():
    """Child ``train`` dict shallowly merges with base ``train`` dict.

    The merge is ONE level deep: top-level dict keys are merged, but
    nested dicts (e.g. ``train.optimizer``) are replaced wholesale by
    the child value (the original ``_load_config`` semantics).

    This test documents the CORRECT existing behaviour so that the
    extracted ``gp1.config.load_config`` is semantically identical.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_yaml(
            tmp_path / "base.yaml",
            {
                "train": {
                    "max_epochs": 50,
                    "optimizer": {
                        "name": "adamw",
                        "lr": 1e-3,
                        "weight_decay": 1e-4,
                    },
                }
            },
        )
        _write_yaml(
            tmp_path / "child.yaml",
            {
                "defaults": ["base"],
                # Child supplies a NEW optimizer sub-dict — replaces base's.
                "train": {
                    "max_epochs": 100,
                    "optimizer": {"name": "novograd", "lr": 2e-4},
                },
            },
        )
        result = load_config(tmp_path / "child.yaml")

    # top-level key max_epochs comes from the shallow merge of train dicts.
    assert result["train"]["max_epochs"] == 100
    # optimizer is replaced wholesale (one-level-deep merge semantics).
    opt = result["train"]["optimizer"]
    assert opt["lr"] == pytest.approx(2e-4), "child lr wins"
    assert opt["name"] == "novograd", "child optimizer name wins"
    # weight_decay was NOT in the child optimizer dict — it is gone
    # (one-level semantics: nested dicts replace, not merge).
    assert "weight_decay" not in opt, (
        "weight_decay must be absent — child optimizer replaces base optimizer"
    )


# ---------------------------------------------------------------------------
# Test 4: scripts/train.py no longer defines its own _load_config
# ---------------------------------------------------------------------------


def test_scripts_train_uses_shared_load_config():
    """scripts/train.py must not define _load_config locally — it should either
    re-export gp1.config.load_config or reference it directly."""
    import train as train_script  # noqa: PLC0415 — intentional late import

    # Either _load_config is gp1.config.load_config itself (same object),
    # or _load_config is not defined (train.py calls load_config directly).
    if hasattr(train_script, "_load_config"):
        # If it exists, it must be the same callable as gp1.config.load_config.
        assert train_script._load_config is load_config, (
            "train._load_config must be gp1.config.load_config, not a local duplicate"
        )


# ---------------------------------------------------------------------------
# Test 5: real quartznet_10x4.yaml round-trip matches expected keys
# ---------------------------------------------------------------------------


def test_load_config_real_quartznet_config():
    """load_config on the real quartznet_10x4.yaml must resolve defaults and
    produce the expected top-level keys."""
    cfg_path = _CONFIGS_DIR / "quartznet_10x4.yaml"
    if not cfg_path.exists():
        pytest.skip("quartznet_10x4.yaml not found")
    result = load_config(cfg_path)
    assert "model" in result
    assert "train" in result
    assert "audio" in result
    assert result["model"]["name"] == "quartznet_10x4"
    assert "defaults" not in result
