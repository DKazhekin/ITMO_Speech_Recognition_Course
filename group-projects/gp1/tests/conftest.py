"""Shared pytest setup for GP1 tests.

No pyproject.toml is installed in editable mode yet, so we prepend the
project's `src/` directory onto sys.path once per test session. This
makes `from gp1.models.common import TCSConvBlock` etc. importable when
pytest is launched from the gp1 project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
