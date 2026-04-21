"""End-to-end smoke test for train → predict CLI orchestration.

Generates a tiny synthetic dataset (random-noise WAVs + digit transcriptions
in 1000..9999), writes Kaggle-style CSVs, runs ``scripts/train.py`` and
``scripts/predict.py`` as subprocesses on CPU, and verifies the submission
CSV has the expected columns and row count.

Marked ``slow``: ~6-10s on CPU. Skipped if the project ``.venv`` is absent.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"

_SAMPLE_RATE = 16000
_DURATION_SEC = 3.0
_N_TRAIN = 8
_N_DEV = 4
_N_TEST = 4
_DIGIT_RANGE = (1000, 9999)


def _python_bin() -> str:
    """Prefer the project's .venv; fall back to the pytest runner's python."""
    if _VENV_PYTHON.exists():
        return str(_VENV_PYTHON)
    return sys.executable


def _generate_wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_samples = int(_SAMPLE_RATE * _DURATION_SEC)
    samples = (rng.standard_normal(n_samples) * 0.05).astype(np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, subtype="FLOAT")


def _write_split(
    rows: list[dict], audio_dir: Path, csv_path: Path, seed_offset: int
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        _generate_wav(audio_dir / row["filename"], seed=seed_offset + i)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "transcription", "spk_id", "gender"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_rows(n: int, rng: random.Random, prefix: str) -> list[dict]:
    return [
        {
            "filename": f"{prefix}_{i:03d}.wav",
            "transcription": str(rng.randint(*_DIGIT_RANGE)),
            "spk_id": f"spk_{i % 2}",
            "gender": "m" if i % 2 == 0 else "f",
        }
        for i in range(n)
    ]


def _write_smoke_config(path: Path) -> None:
    config = {
        "audio": {
            "samplerate": _SAMPLE_RATE,
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "n_mels": 80,
        },
        "aug": {
            "speed_factors": [1.0],
            "speed_prob": 0.0,
            "vtlp_prob": 0.0,
            "pitch_prob": 0.0,
            "gain_prob": 0.0,
            "noise_prob": 0.0,
            "rir_prob": 0.0,
        },
        "data": {"train_batch_size": 2, "dev_batch_size": 2},
        "model": {"name": "quartznet_10x4", "d_model": 256, "dropout": 0.1},
        "train": {
            "optimizer": {
                "name": "adamw",
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 2,
                "min_lr_ratio": 0.1,
            },
            "grad_accum": 1,
            "fp16_autocast": False,
            "max_epochs": 1,
            "log_every_n_steps": 1,
            "val_every_n_epochs": 1,
            "early_stop_patience": 5,
            "early_stop_metric": "max_speaker_cer",
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed (rc={result.returncode}):\n"
            f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.slow
def test_train_then_predict_produces_valid_submission(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    if not shutil.which(_python_bin()):
        pytest.skip("no usable python interpreter for subprocess")

    rng = random.Random(1337)
    train_rows = _make_rows(_N_TRAIN, rng, "train")
    dev_rows = _make_rows(_N_DEV, rng, "dev")
    test_rows = _make_rows(_N_TEST, rng, "test")

    _write_split(train_rows, tmp_path / "train", tmp_path / "train.csv", 0)
    _write_split(dev_rows, tmp_path / "dev", tmp_path / "dev.csv", 1000)
    _write_split(test_rows, tmp_path / "test", tmp_path / "test.csv", 2000)

    config_path = tmp_path / "smoke.yaml"
    _write_smoke_config(config_path)

    output_dir = tmp_path / "run"
    python = _python_bin()

    _run(
        [
            python,
            str(_SCRIPTS / "train.py"),
            "--config",
            str(config_path),
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--train-root",
            str(tmp_path / "train"),
            "--dev-csv",
            str(tmp_path / "dev.csv"),
            "--dev-root",
            str(tmp_path / "dev"),
            "--output-dir",
            str(output_dir),
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "1337",
        ]
    )

    result_json = output_dir / "result.json"
    assert result_json.exists(), f"result.json missing at {result_json}"
    result = json.loads(result_json.read_text(encoding="utf-8"))
    checkpoint = Path(result["best_ckpt_path"])
    assert checkpoint.exists(), f"best checkpoint missing at {checkpoint}"

    submission = tmp_path / "submission.csv"
    _run(
        [
            python,
            str(_SCRIPTS / "predict.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--test-csv",
            str(tmp_path / "test.csv"),
            "--test-root",
            str(tmp_path / "test"),
            "--output",
            str(submission),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert submission.exists(), "submission.csv not created"
    with open(submission, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["filename", "transcription"]
    assert len(rows) == _N_TEST + 1

    for filename, _transcription in rows[1:]:
        assert filename.endswith(".wav"), f"bad filename in submission: {filename!r}"
