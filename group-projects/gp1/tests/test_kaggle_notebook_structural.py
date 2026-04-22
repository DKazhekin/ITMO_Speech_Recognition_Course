"""Structural tests for src/gp1/submit/kaggle_notebook.ipynb — RED phase.

Validates the notebook JSON structure without executing any cells:
1. Parses with nbformat.read without error.
2. Has at least one markdown cell with "Submission" (case-insensitive).
3. Has at least one code cell that references run_inference or imports from gp1.submit.inference.
4. Has a code cell that writes submission.csv to /kaggle/working/submission.csv.
5. Has a code cell that downloads a release asset from GitHub.
6. Has NO cell that calls scripts/train.py (inference-only notebook).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "gp1"
    / "submit"
    / "kaggle_notebook.ipynb"
)


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    """Parse and return the notebook node; fail clearly if file is missing."""
    assert _NOTEBOOK_PATH.exists(), (
        f"Notebook not found at {_NOTEBOOK_PATH}. Run Phase 2 (GREEN) to create it."
    )
    with open(_NOTEBOOK_PATH, encoding="utf-8") as fh:
        return nbformat.read(fh, as_version=4)


def _all_source(nb: nbformat.NotebookNode) -> list[str]:
    """Return source text of every cell as a list."""
    return [cell.get("source", "") for cell in nb.cells]


def _code_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [cell.get("source", "") for cell in nb.cells if cell.cell_type == "code"]


def _markdown_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [cell.get("source", "") for cell in nb.cells if cell.cell_type == "markdown"]


# ---------------------------------------------------------------------------
# Test 1 – parses without error
# ---------------------------------------------------------------------------


def test_notebook_parses_without_error(notebook: nbformat.NotebookNode) -> None:
    assert notebook is not None
    assert len(notebook.cells) > 0


# ---------------------------------------------------------------------------
# Test 2 – markdown cell with "Submission"
# ---------------------------------------------------------------------------


def test_notebook_has_submission_markdown(notebook: nbformat.NotebookNode) -> None:
    sources = _markdown_sources(notebook)
    assert any("submission" in src.lower() for src in sources), (
        "No markdown cell containing 'Submission' (case-insensitive) found."
    )


# ---------------------------------------------------------------------------
# Test 3 – code cell references run_inference
# ---------------------------------------------------------------------------


def test_notebook_references_run_inference(notebook: nbformat.NotebookNode) -> None:
    code_sources = _code_sources(notebook)
    assert any(
        "run_inference" in src or "gp1.submit.inference" in src for src in code_sources
    ), "No code cell references run_inference or imports from gp1.submit.inference."


# ---------------------------------------------------------------------------
# Test 4 – writes submission.csv to /kaggle/working/submission.csv
# ---------------------------------------------------------------------------


def test_notebook_writes_submission_csv(notebook: nbformat.NotebookNode) -> None:
    code_sources = _code_sources(notebook)
    assert any("/kaggle/working/submission.csv" in src for src in code_sources), (
        "No code cell writes to /kaggle/working/submission.csv."
    )


# ---------------------------------------------------------------------------
# Test 5 – downloads a release asset from GitHub
# ---------------------------------------------------------------------------


def test_notebook_downloads_github_release_asset(
    notebook: nbformat.NotebookNode,
) -> None:
    code_sources = _code_sources(notebook)
    has_download = any(
        "github.com" in src
        and "releases/download" in src
        or "gh release download" in src
        for src in code_sources
    )
    assert has_download, (
        "No code cell downloads a GitHub release asset "
        "(expected 'github.com/.../releases/download/' or 'gh release download')."
    )


# ---------------------------------------------------------------------------
# Test 6 – no cell calls scripts/train.py (must be inference-only)
# ---------------------------------------------------------------------------


def test_notebook_does_not_call_train_script(
    notebook: nbformat.NotebookNode,
) -> None:
    all_sources = _all_source(notebook)
    assert not any("scripts/train.py" in src for src in all_sources), (
        "Notebook must NOT call scripts/train.py — it is an inference-only notebook."
    )
