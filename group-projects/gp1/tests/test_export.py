"""Tests for scripts/export.py — RED phase.

Tests verify the CLI behaviour of the export script:
- creates release directory with required files
- optional LM binary copy
- --slim strips optimizer state
- release.json metadata written
- rejects missing checkpoint
- rejects existing dir without --force
- zero exit code on success
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "export.py"
_PYTHON = sys.executable


def _make_fake_checkpoint(tmp_path: Path, best_val_cer: float = 0.018) -> Path:
    """Write a minimal checkpoint that matches trainer.py payload format."""
    # Minimal state_dict with a real parameter tensor so params_count > 0
    state_dict = {"layer.weight": torch.zeros(4, 4)}
    payload = {
        "model": state_dict,
        "optimizer": {"state": {}, "param_groups": []},
        "step": 100,
        "epoch": 42,
        "best_val_cer": best_val_cer,
        "config": {"max_epochs": 50, "grad_accum": 1},
    }
    ckpt_path = tmp_path / "epoch0042_cer0.0180.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def _make_fake_config(tmp_path: Path) -> Path:
    """Write a minimal YAML config file."""
    cfg_path = tmp_path / "quartznet_10x4.yaml"
    cfg_path.write_text("model:\n  name: quartznet\n  blocks: 10\n")
    return cfg_path


def _run_export(
    *extra_args: str,
    ckpt: Path,
    config: Path,
    baseline: str = "quartznet",
    tag: str = "v0.1.0-test",
    output_root: Path | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        _PYTHON,
        str(_EXPORT_SCRIPT),
        "--checkpoint",
        str(ckpt),
        "--config",
        str(config),
        "--baseline",
        baseline,
        "--tag",
        tag,
    ]
    if output_root is not None:
        cmd += ["--output-root", str(output_root)]
    cmd += list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Test 1 – required files are created
# ---------------------------------------------------------------------------


def test_export_creates_release_dir_with_required_files(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, result.stderr

    release_dir = output_root / "quartznet" / "v0.1.0-test"
    assert release_dir.is_dir()

    # model.pt must exist and contain the "model" key
    model_pt = release_dir / "model.pt"
    assert model_pt.exists()
    loaded = torch.load(model_pt, weights_only=False)
    assert "model" in loaded

    assert (release_dir / "config.yaml").exists()
    assert (release_dir / "README.md").exists()


# ---------------------------------------------------------------------------
# Test 2 – optional LM binary
# ---------------------------------------------------------------------------


def test_export_optional_lm_is_copied_when_provided(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    lm_file = tmp_path / "lm.bin"
    lm_file.write_bytes(b"\x00" * 8)  # fake binary

    result = _run_export(
        "--lm-binary",
        str(lm_file),
        ckpt=ckpt,
        config=cfg,
        output_root=output_root,
    )
    assert result.returncode == 0, result.stderr
    release_dir = output_root / "quartznet" / "v0.1.0-test"
    assert (release_dir / "lm.bin").exists()


def test_export_lm_not_copied_when_not_provided(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, result.stderr
    release_dir = output_root / "quartznet" / "v0.1.0-test"
    assert not (release_dir / "lm.bin").exists()


# ---------------------------------------------------------------------------
# Test 3 – --slim strips optimizer state
# ---------------------------------------------------------------------------


def test_export_strips_optimizer_state_when_flag_set(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export("--slim", ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, result.stderr

    release_dir = output_root / "quartznet" / "v0.1.0-test"
    loaded = torch.load(release_dir / "model.pt", weights_only=False)
    assert "model" in loaded
    assert "optimizer" not in loaded


def test_export_without_slim_keeps_optimizer_state(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, result.stderr

    release_dir = output_root / "quartznet" / "v0.1.0-test"
    loaded = torch.load(release_dir / "model.pt", weights_only=False)
    assert "optimizer" in loaded


# ---------------------------------------------------------------------------
# Test 4 – release.json metadata
# ---------------------------------------------------------------------------


def test_export_writes_release_metadata_json(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path, best_val_cer=0.0123)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, result.stderr

    release_dir = output_root / "quartznet" / "v0.1.0-test"
    meta_path = release_dir / "release.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["baseline"] == "quartznet"
    assert meta["tag"] == "v0.1.0-test"
    assert "git_commit" in meta
    assert "checkpoint_source" in meta
    assert "params_count" in meta
    assert isinstance(meta["params_count"], int)
    assert meta["params_count"] > 0
    assert "best_val_cer" in meta
    assert abs(meta["best_val_cer"] - 0.0123) < 1e-6


# ---------------------------------------------------------------------------
# Test 5 – rejects missing checkpoint
# ---------------------------------------------------------------------------


def test_export_rejects_missing_checkpoint(tmp_path: Path) -> None:
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(
        ckpt=tmp_path / "does_not_exist.pt",
        config=cfg,
        output_root=output_root,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Test 6 – reject existing dir without --force, overwrite with --force
# ---------------------------------------------------------------------------


def test_export_rejects_existing_release_dir_without_force(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    # First run succeeds
    r1 = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert r1.returncode == 0, r1.stderr

    # Second run WITHOUT --force must fail
    r2 = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert r2.returncode != 0


def test_export_force_overwrites_existing_release_dir(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    r1 = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert r1.returncode == 0, r1.stderr

    r2 = _run_export("--force", ckpt=ckpt, config=cfg, output_root=output_root)
    assert r2.returncode == 0, r2.stderr


# ---------------------------------------------------------------------------
# Test 7 – zero exit code on success (redundant but explicit)
# ---------------------------------------------------------------------------


def test_export_cli_exit_code_zero_on_success(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    cfg = _make_fake_config(tmp_path)
    output_root = tmp_path / "releases"

    result = _run_export(ckpt=ckpt, config=cfg, output_root=output_root)
    assert result.returncode == 0, f"stderr: {result.stderr}"
