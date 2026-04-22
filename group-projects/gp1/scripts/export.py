"""Export a trained GP1 checkpoint to a release directory.

Packs checkpoint, config, optional LM binary, and metadata into
``releases/<baseline>/<tag>/`` for publishing via
``scripts/publish_release.sh``.

Usage::

    python scripts/export.py \\
        --checkpoint runs/quartznet_baseline/checkpoints/epoch0042_cer0.0180.pt \\
        --config configs/quartznet_10x4.yaml \\
        --baseline quartznet \\
        --tag v0.1.0 \\
        --output-root releases \\
        [--lm-binary data/lm/lm.bin] \\
        [--slim] \\
        [--force]

See also:
  - gh release create docs: https://cli.github.com/manual/gh_release_create
  - CONTRACTS.md §8 for checkpoint payload format
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("gp1.export")

# Use shared config loader from gp1.config to resolve `defaults:` inheritance.
try:
    from gp1.config import load_config as _resolve_config  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    _resolve_config = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_release_dir(output_root: Path, baseline: str, tag: str) -> Path:
    """Return ``output_root/<baseline>/<tag>`` without creating it."""
    return output_root / baseline / tag


def _git_commit_hash() -> str:
    """Return the current HEAD commit hash (or 'unknown' on failure)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _count_params(state_dict: dict) -> int:
    """Count floating-point parameters in a state dict."""
    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
            total += tensor.numel()
    return total


def _write_release_metadata(
    release_dir: Path,
    *,
    baseline: str,
    tag: str,
    checkpoint_source: str,
    params_count: int,
    best_val_cer: float,
) -> None:
    """Write release.json with release metadata.

    Args:
        release_dir: Directory where release.json will be written.
        baseline: Model baseline name (e.g. "quartznet").
        tag: Release tag (e.g. "v0.1.0").
        checkpoint_source: Basename of the source checkpoint file.
        params_count: Number of floating-point parameters.
        best_val_cer: Best validation CER from training.
    """
    git_commit = _git_commit_hash()
    # Convert NaN/inf to JSON null so release.json is RFC 8259 compliant
    # (jq and Node.JSON.parse reject the non-standard Infinity literal).
    cer_for_json = (
        None if (math.isinf(best_val_cer) or math.isnan(best_val_cer)) else best_val_cer
    )
    metadata = {
        "baseline": baseline,
        "tag": tag,
        "git_commit": git_commit,
        "checkpoint_source": checkpoint_source,
        "params_count": params_count,
        "best_val_cer": cer_for_json,
    }
    meta_path = release_dir / "release.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Wrote release metadata to %s", meta_path)


def _write_readme(
    release_dir: Path,
    *,
    baseline: str,
    tag: str,
    params_count: int,
    best_val_cer: float,
) -> None:
    """Write README.md with a short summary for the release.

    References:
        - Kaggle submission docs: https://www.kaggle.com/docs/competitions#kernels
        - gh release create: https://cli.github.com/manual/gh_release_create
    """
    readme = f"""# GP1 ASR Release — {baseline} {tag}

## Summary

| Field | Value |
|---|---|
| Baseline | {baseline} |
| Tag | {tag} |
| Parameters | {params_count:,} |
| Best val CER | {best_val_cer:.4f} |

## Loading the model

```python
import torch
from pathlib import Path
from gp1.submit.inference import InferenceConfig, run_inference

cfg = InferenceConfig(
    checkpoint_path=Path("model.pt"),
    config_path=Path("config.yaml"),
    lm_binary_path=Path("lm.bin") if Path("lm.bin").exists() else None,
    batch_size=32,
    device="cpu",
)
```

## Running inference

```bash
python scripts/predict.py \\
    --checkpoint model.pt \\
    --config config.yaml \\
    --test-csv test.csv \\
    --test-root /data/test/ \\
    --output submission.csv
```

## Kaggle submission

Use `src/gp1/submit/kaggle_notebook.ipynb`. Set `RELEASE_TAG = "{tag}"` in
the configuration cell, then run all cells.
"""
    readme_path = release_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    log.info("Wrote README.md to %s", readme_path)


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------


def export(
    checkpoint_path: Path,
    config_path: Path,
    baseline: str,
    tag: str,
    output_root: Path,
    lm_binary_path: Path | None = None,
    slim: bool = False,
    force: bool = False,
) -> Path:
    """Export a checkpoint to a release directory.

    Args:
        checkpoint_path: Path to the source .pt checkpoint.
        config_path: Path to the YAML model config.
        baseline: Name of the model baseline (e.g. "quartznet").
        tag: Release tag string (e.g. "v0.1.0").
        output_root: Root directory where releases are written.
        lm_binary_path: Optional path to a KenLM binary to bundle.
        slim: If True, strip the optimizer state from the checkpoint.
        force: If True, overwrite an existing release directory.

    Returns:
        Path to the created release directory.

    Raises:
        SystemExit: On any unrecoverable error (missing files, dir exists).
    """
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        raise SystemExit(1)

    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        raise SystemExit(1)

    if lm_binary_path is not None and not lm_binary_path.exists():
        log.error("LM binary not found: %s", lm_binary_path)
        raise SystemExit(1)

    release_dir = _resolve_release_dir(output_root, baseline, tag)

    if release_dir.exists():
        if not force:
            log.error(
                "Release directory already exists: %s  (use --force to overwrite)",
                release_dir,
            )
            raise SystemExit(1)
        log.info("--force: removing existing release dir %s", release_dir)
        shutil.rmtree(release_dir)

    release_dir.mkdir(parents=True, exist_ok=False)
    log.info("Created release directory: %s", release_dir)

    # ------------------------------------------------------------------ ckpt
    log.info("Loading checkpoint from %s", checkpoint_path)
    payload: dict = torch.load(checkpoint_path, weights_only=False)

    if slim:
        payload = {
            "model": payload["model"],
            "epoch": payload.get("epoch", -1),
            "step": payload.get("step", -1),
            "best_val_cer": payload.get("best_val_cer", float("nan")),
            "config": payload.get("config", {}),
        }
        log.info("--slim: optimizer state stripped from checkpoint")

    model_pt_path = release_dir / "model.pt"
    torch.save(payload, model_pt_path)
    log.info("Wrote model.pt (%d bytes)", model_pt_path.stat().st_size)

    # ----------------------------------------------------------------- config
    # H9: resolve `defaults:` inheritance before writing config.yaml so the
    # released config is self-contained (no relative `defaults:` references).
    config_dst = release_dir / "config.yaml"
    if _resolve_config is not None:
        resolved_cfg = _resolve_config(config_path)
        config_dst.write_text(
            yaml.dump(resolved_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        log.info("Wrote resolved config.yaml from %s", config_path)
    else:
        shutil.copy2(config_path, config_dst)
        log.info("Copied config.yaml (resolution unavailable) from %s", config_path)

    # -------------------------------------------------------------------- lm
    if lm_binary_path is not None:
        shutil.copy2(lm_binary_path, release_dir / "lm.bin")
        log.info("Copied lm.bin from %s", lm_binary_path)

    # --------------------------------------------------------------- metadata
    state_dict = payload["model"]
    params_count = _count_params(state_dict)
    best_val_cer: float = float(payload.get("best_val_cer", float("nan")))

    _write_release_metadata(
        release_dir,
        baseline=baseline,
        tag=tag,
        checkpoint_source=checkpoint_path.name,
        params_count=params_count,
        best_val_cer=best_val_cer,
    )

    _write_readme(
        release_dir,
        baseline=baseline,
        tag=tag,
        params_count=params_count,
        best_val_cer=best_val_cer,
    )

    log.info("Export complete: %s", release_dir)
    return release_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a GP1 checkpoint to a release directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the source .pt checkpoint file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML model config.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=str,
        help='Model baseline name, e.g. "quartznet".',
    )
    parser.add_argument(
        "--tag",
        required=True,
        type=str,
        help='Release tag, e.g. "v0.1.0".',
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("releases"),
        help="Root directory for releases (default: ./releases).",
    )
    parser.add_argument(
        "--lm-binary",
        type=Path,
        default=None,
        help="Optional path to a KenLM binary to bundle as lm.bin.",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        default=False,
        help="Strip optimizer state from the exported checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing release directory.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        baseline=args.baseline,
        tag=args.tag,
        output_root=args.output_root,
        lm_binary_path=args.lm_binary,
        slim=args.slim,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
