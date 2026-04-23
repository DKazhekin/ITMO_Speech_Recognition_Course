"""Unit tests for gp1.env — platform detection and environment setup."""

import dataclasses
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gp1 import env


@pytest.mark.unit
def test_detect_platform_returns_local_when_no_indicators(monkeypatch):
    """No Kaggle/Colab signals -> platform is 'local'."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", None)

    assert env.detect_platform() == "local"


@pytest.mark.unit
def test_detect_platform_returns_kaggle_when_env_set(monkeypatch):
    """KAGGLE_KERNEL_RUN_TYPE set -> platform is 'kaggle'."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", None)

    assert env.detect_platform() == "kaggle"


@pytest.mark.unit
def test_detect_platform_returns_colab_when_env_set(monkeypatch):
    """COLAB_GPU set -> platform is 'colab'."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("COLAB_GPU", "1")
    monkeypatch.setitem(sys.modules, "google.colab", None)

    assert env.detect_platform() == "colab"


@pytest.mark.unit
def test_install_platform_deps_local_is_noop():
    """install_platform_deps('local') must not invoke subprocess."""
    with patch("subprocess.run") as mock_run:
        env.install_platform_deps("local")
        mock_run.assert_not_called()


@pytest.mark.unit
def test_setup_environment_raises_when_paths_missing(monkeypatch, tmp_path):
    """setup_environment raises RuntimeError when data roots do not exist."""
    monkeypatch.setattr(env, "detect_platform", lambda: "local")
    monkeypatch.setattr(env, "install_platform_deps", lambda _: None)

    fake_paths = env.Paths(
        repo_root=tmp_path,
        train_root=tmp_path / "nonexistent_train",
        dev_root=tmp_path / "nonexistent_dev",
        test_root=None,
        train_csv=tmp_path / "train.csv",
        dev_csv=tmp_path / "dev.csv",
        ckpt_root=tmp_path / "checkpoints",
    )
    monkeypatch.setattr(env, "_local_paths", lambda: fake_paths)

    with pytest.raises(RuntimeError, match="Expected path does not exist"):
        env.setup_environment()


@pytest.mark.unit
def test_paths_is_frozen_dataclass():
    """Paths must be a frozen dataclass — mutation raises FrozenInstanceError."""
    paths = env.Paths(
        repo_root=Path("/tmp"),
        train_root=Path("/tmp/train"),
        dev_root=Path("/tmp/dev"),
        test_root=None,
        train_csv=Path("/tmp/train.csv"),
        dev_csv=Path("/tmp/dev.csv"),
        ckpt_root=Path("/tmp/ckpts"),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.repo_root = Path("/other")  # type: ignore[misc]
