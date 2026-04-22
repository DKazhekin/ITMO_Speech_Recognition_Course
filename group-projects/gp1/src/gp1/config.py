"""Shared YAML config loading with `defaults:` inheritance resolution.

Single source of truth for config loading across scripts/train.py,
scripts/export.py, scripts/smoke_test_all_configs.py, and notebooks.

The ``load_config`` function is semantically identical to the original
``_load_config`` in scripts/train.py (lines 74-100). Any change to
inheritance semantics must be made here and propagated via this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config and recursively resolve ``defaults:`` inheritance.

    Each name under ``defaults`` is resolved as ``<path.parent>/<name>.yaml``
    and merged in order: parent (defaults) configs are loaded first, then
    the current file's top-level keys override.  Dict values are merged one
    level deep; all other types are replaced wholesale.

    Supports multi-parent defaults::

        defaults:
          - base
          - extras

    Child key overrides win over all parents.  Parent keys are applied in
    list order (later parents win over earlier parents before the child
    applies its overrides).

    Args:
        path: Path to the YAML config file to load.

    Returns:
        Fully-resolved config dict with no ``defaults`` key.
    """
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg
    merged: dict[str, Any] = {}
    for name in defaults:
        parent = load_config(path.parent / f"{name}.yaml")
        for k, v in parent.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged
