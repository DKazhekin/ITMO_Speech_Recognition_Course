"""Checkpoint save/load — pure state_dict + meta.json sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def save_best(model: nn.Module, meta: dict, ckpt_dir: Path) -> Path:
    """Save model state_dict to ckpt_dir/best.pt and meta to ckpt_dir/meta.json.

    Unwraps ``torch.compile``'s OptimizedModule via ``_orig_mod`` so the saved
    keys match a non-compiled instance of the same architecture at load time.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt_path = ckpt_dir / "best.pt"
    inner = getattr(model, "_orig_mod", model)
    torch.save(inner.state_dict(), pt_path)
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return pt_path


def load_checkpoint(ckpt_path: Path, model: nn.Module) -> dict:
    """Load state_dict into model from ckpt_path; return parsed meta.json beside it."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    meta = json.loads((ckpt_path.parent / "meta.json").read_text())
    return meta
