"""Inference entrypoint for GP1 Russian spoken-numbers ASR.

Reads a Kaggle test CSV (or an existing JSONL manifest), runs
``gp1.submit.inference.run_inference`` over it, and writes a Kaggle
submission CSV with columns ``filename,transcription``.

Usage:

    python scripts/predict.py \\
        --checkpoint runs/quartznet_baseline/checkpoints/best.pt \\
        --config configs/quartznet_10x4.yaml \\
        --test-csv /path/to/test.csv --test-root /path/to/test/ \\
        --output submission.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch

from gp1.data.manifest import build_manifest, read_jsonl
from gp1.submit.inference import InferenceConfig, run_inference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("gp1.predict")


def main() -> int:
    parser = argparse.ArgumentParser(description="GP1 ASR inference → submission.csv")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument("--lm-binary", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # -------------------------------------------------------------- manifest
    if args.test_manifest is not None:
        manifest_path = args.test_manifest
    else:
        if args.test_csv is None or args.test_root is None:
            parser.error("Provide --test-manifest OR both --test-csv and --test-root")
        manifest_path = args.output.with_suffix(".manifest.jsonl")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        build_manifest(args.test_csv, args.test_root, manifest_path)

    records = read_jsonl(manifest_path)
    log.info("Loaded %d test records from %s", len(records), manifest_path)

    # ------------------------------------------------------------- inference
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        lm_binary_path=args.lm_binary,
        batch_size=args.batch_size,
        device=device,
    )
    predictions = run_inference(records, config)

    # ------------------------------------------------------------- write csv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "transcription"])
        for filename, digits in predictions:
            writer.writerow([filename, digits])
    log.info("Wrote %d predictions to %s", len(predictions), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
