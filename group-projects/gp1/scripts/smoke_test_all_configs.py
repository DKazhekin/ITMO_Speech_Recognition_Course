"""Synthetic smoke-test for all GP1 architectures.

For each config in configs/*.yaml (except base.yaml), we:
  1. Generate a handful of zero-padded synthetic WAV files with short,
     valid transcriptions.
  2. Write a CSV in the Kaggle format expected by gp1.data.manifest.build_manifest.
  3. Run scripts/train.py for one mini-epoch on this dummy data.
  4. Report pass/fail per config.

This verifies that the full train-pipeline (data → model → losses → trainer)
wires together without crashing. It does NOT verify learning.

Usage:
  .venv/bin/python scripts/smoke_test_all_configs.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoke")


# Six short, valid 4-6 digit transcriptions for the synthetic dataset.
_TRANSCRIPTIONS = ["100000", "259341", "700005", "123456", "987654", "400000"]
_N_TRAIN = 8
_N_DEV = 4


def _make_dummy_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> None:
    n = int(duration_s * sr)
    audio = (
        np.random.RandomState(hash(str(path)) & 0xFFFFFFFF).randn(n).astype(np.float32)
        * 0.01
    )
    sf.write(str(path), audio, sr)


def _write_csv(csv_path: Path, wav_dir: Path, n: int, seed: int) -> list[str]:
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    names: list[str] = []
    for i in range(n):
        trans = _TRANSCRIPTIONS[rng.randint(len(_TRANSCRIPTIONS))]
        fname = f"sample_{seed}_{i:03d}.wav"
        wav_path = wav_dir / fname
        _make_dummy_wav(wav_path)
        rows.append(
            {
                "filename": fname,
                "transcription": trans,
                "spk_id": f"spk_{i % 3:02d}",
                "gender": "male" if i % 2 == 0 else "female",
            }
        )
        names.append(fname)

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "transcription", "spk_id", "gender"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return names


def _patch_config(config_path: Path, out_path: Path) -> Path:
    """Load a config, flatten its `defaults: [base]` into a single dict, and
    override training knobs for a fast smoke run.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve `defaults: [base]` manually (the script does this too).
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        for name in defaults:
            base_path = config_path.parent / f"{name}.yaml"
            with open(base_path, encoding="utf-8") as fh:
                base_cfg = yaml.safe_load(fh)
            # Merge: base first, then override with cfg
            merged: dict = {}
            for key, val in base_cfg.items():
                merged[key] = val
            for key, val in cfg.items():
                if isinstance(val, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            cfg = merged

    # Override training knobs for a fast run.
    cfg.setdefault("train", {})
    cfg["train"]["max_epochs"] = 1
    cfg["train"]["fp16_autocast"] = False
    cfg["train"]["grad_accum"] = 1
    cfg["train"]["early_stop_patience"] = 100
    cfg["train"].setdefault("optimizer", {})
    cfg["train"].setdefault("scheduler", {})
    cfg["train"]["scheduler"]["warmup_steps"] = 1

    # Use batch_size instead of max_tokens_per_batch for predictability.
    cfg.setdefault("data", {})
    cfg["data"]["max_tokens_per_batch"] = None
    cfg["data"]["train_batch_size"] = 2
    cfg["data"]["dev_batch_size"] = 2

    # For BPE configs, train a tiny SentencePiece model on the synthetic
    # corpus and patch bpe_model_path to point at it. The model_path is
    # resolved by scripts/train.py relative to the config file's parent,
    # so we write the model next to the patched config.
    text_cfg = cfg.get("text", {}) or {}
    if text_cfg.get("vocab_type") == "bpe":
        bpe_model_path = _train_tiny_bpe(out_path.parent, vocab_size=64)
        cfg.setdefault("text", {})
        cfg["text"]["bpe_model_path"] = str(bpe_model_path.name)
        # The model head must match the actual BPE vocab size (64 + 1 blank).
        cfg.setdefault("model", {})
        cfg["model"]["vocab_size"] = 65

    out_path.write_text(yaml.safe_dump(cfg))
    return out_path


def _train_tiny_bpe(out_dir: Path, vocab_size: int = 64) -> Path:
    """Train a minimal SentencePiece BPE model for the smoke-test."""
    from gp1.text.normalize import digits_to_words
    from gp1.text.vocab_bpe import train_bpe_model

    corpus = out_dir / "bpe_corpus.txt"
    lines = [digits_to_words(t) for t in _TRANSCRIPTIONS]
    # Repeat a few times so SP has enough material to fit vocab_size.
    corpus.write_text("\n".join(lines * 20) + "\n", encoding="utf-8")
    model_prefix = out_dir / "tiny_bpe"
    return train_bpe_model(
        corpus_path=corpus,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
    )


def _run_train(cfg_path: Path, run_dir: Path, data_dir: Path) -> tuple[int, str]:
    train_csv = data_dir / "train.csv"
    dev_csv = data_dir / "dev.csv"
    wav_dir = data_dir / "wavs"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(cfg_path),
        "--train-csv",
        str(train_csv),
        "--train-root",
        str(wav_dir),
        "--dev-csv",
        str(dev_csv),
        "--dev-root",
        str(wav_dir),
        "--output-dir",
        str(run_dir),
        "--num-workers",
        "0",
        "--device",
        "cpu",
    ]
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        return result.returncode, tail
    return 0, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of config basenames to test (default: all non-base)",
    )
    args = parser.parse_args()

    configs_dir = ROOT / "configs"
    all_configs = [
        p for p in sorted(configs_dir.glob("*.yaml")) if p.stem not in {"base"}
    ]
    if args.configs:
        keep = set(args.configs)
        all_configs = [p for p in all_configs if p.stem in keep]

    if not all_configs:
        log.error("No configs matched")
        return 1

    with tempfile.TemporaryDirectory() as tmp_root_str:
        tmp_root = Path(tmp_root_str)

        # Shared synthetic data — write ONCE, reuse across all configs.
        data_dir = tmp_root / "data"
        wav_dir = data_dir / "wavs"
        wav_dir.mkdir(parents=True)
        _write_csv(data_dir / "train.csv", wav_dir, n=_N_TRAIN, seed=1)
        _write_csv(data_dir / "dev.csv", wav_dir, n=_N_DEV, seed=2)

        results: list[tuple[str, int, str]] = []
        for cfg_path in all_configs:
            log.info("===== Testing config: %s =====", cfg_path.name)
            run_dir = tmp_root / f"run_{cfg_path.stem}"
            run_dir.mkdir()
            patched_cfg = _patch_config(cfg_path, run_dir / "patched_config.yaml")

            rc, tail = _run_train(patched_cfg, run_dir, data_dir)
            results.append((cfg_path.stem, rc, tail))

        # Summary
        log.info("===== Smoke-test summary =====")
        all_pass = True
        for name, rc, tail in results:
            status = "PASS" if rc == 0 else "FAIL"
            log.info("  %s  %s", status, name)
            if rc != 0:
                all_pass = False
                log.error("    last stderr lines:\n%s", tail)
        return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
