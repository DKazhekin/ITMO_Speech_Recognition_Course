"""Bootstrap module: platform detection and environment setup."""

import importlib
import importlib.util
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Dependencies installed on cloud platforms if not already available.
_CLOUD_DEPS = ["sentencepiece", "num2words", "jiwer", "librosa", "tqdm"]


@dataclass(frozen=True)
class Paths:
    """All filesystem roots needed by the ASR training pipeline."""

    repo_root: Path
    train_root: Path
    dev_root: Path
    test_root: Path | None
    train_csv: Path
    dev_csv: Path
    ckpt_root: Path


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platform() -> Literal["local", "colab", "kaggle"]:
    """Return the current runtime platform."""
    # Kaggle: kernel sets this env var
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    # Kaggle: filesystem fallback
    if Path("/kaggle/input").exists():
        return "kaggle"

    # Colab: env var set by the runtime
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    # Colab: google.colab importable
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"

    return "local"


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------


def install_platform_deps(platform: Literal["local", "colab", "kaggle"]) -> None:
    """Install missing cloud dependencies; no-op on local."""
    if platform == "local":
        return

    missing = [pkg for pkg in _CLOUD_DEPS if importlib.util.find_spec(pkg) is None]
    if not missing:
        logger.info("All cloud deps already installed.")
        return

    logger.info("Installing missing deps: %s", missing)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + missing,
        check=True,
    )


# ---------------------------------------------------------------------------
# Per-platform path builders
# ---------------------------------------------------------------------------


def _local_paths() -> Paths:
    """Resolve paths for a local checkout by walking up to the repo root."""
    start = Path.cwd().resolve()
    candidate = start
    while not (candidate / "src" / "gp1" / "__init__.py").exists():
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError(
                "Could not locate gp1 repo root (src/gp1/__init__.py not found). "
                "Run from within the group-projects/gp1 directory tree."
            )
        candidate = parent

    repo_root = candidate
    data_root = repo_root / "data"

    train_root = data_root / "train"
    dev_root = data_root / "dev"
    test_dir = data_root / "test"
    test_root = test_dir if test_dir.exists() else None

    return Paths(
        repo_root=repo_root,
        train_root=train_root,
        dev_root=dev_root,
        test_root=test_root,
        train_csv=train_root / "train.csv",
        dev_csv=dev_root / "dev.csv",
        ckpt_root=repo_root / "checkpoints",
    )


def _colab_paths() -> Paths:
    """Resolve paths for Google Colab (data lives in Google Drive)."""
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        from google.colab import drive  # type: ignore[import]

        drive.mount("/content/drive")

    asr_root = drive_root / "asr-2026"
    train_root = asr_root / "train"
    dev_root = asr_root / "dev"
    test_dir = asr_root / "test"
    test_root = test_dir if test_dir.exists() else None

    return Paths(
        repo_root=Path("/content/ITMO_Speech_Recognition_Course/group-projects/gp1"),
        train_root=train_root,
        dev_root=dev_root,
        test_root=test_root,
        train_csv=train_root / "train.csv",
        dev_csv=dev_root / "dev.csv",
        ckpt_root=asr_root / "checkpoints",
    )


def _kaggle_paths() -> Paths:
    """Resolve paths for Kaggle notebooks (data under /kaggle/input/)."""
    input_root = Path("/kaggle/input")
    candidates = [
        p
        for p in input_root.iterdir()
        if p.is_dir() and (p / "train" / "train.csv").exists()
    ]
    if not candidates:
        raise RuntimeError(
            f"No dataset with train/train.csv found under {input_root}. "
            "Attach a Kaggle Dataset containing train/, dev/, test/ subfolders."
        )
    data_root = candidates[0]

    train_root = data_root / "train"
    dev_root = data_root / "dev"
    test_dir = data_root / "test"
    test_root = test_dir if test_dir.exists() else None

    return Paths(
        repo_root=Path("/kaggle/working"),
        train_root=train_root,
        dev_root=dev_root,
        test_root=test_root,
        train_csv=train_root / "train.csv",
        dev_csv=dev_root / "dev.csv",
        ckpt_root=Path("/kaggle/working/checkpoints"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def setup_environment() -> tuple[Paths, torch.device]:
    """Detect platform, install deps, build paths, return (paths, device)."""
    platform = detect_platform()
    logger.info("Detected platform: %s", platform)

    install_platform_deps(platform)

    _builders = {
        "local": _local_paths,
        "colab": _colab_paths,
        "kaggle": _kaggle_paths,
    }
    if platform not in _builders:
        raise RuntimeError(
            f"Unknown platform '{platform}': set GP1_PATHS env var or edit env.py"
        )
    paths = _builders[platform]()

    for root in (paths.train_root, paths.dev_root):
        if not root.exists():
            raise RuntimeError(f"Expected path does not exist on {platform}: {root}")

    paths.ckpt_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    return paths, device
