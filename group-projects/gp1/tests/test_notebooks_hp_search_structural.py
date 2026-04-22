"""Structural tests for HP-search notebooks (RED phase).

Validates run_kaggle.ipynb, run_colab.ipynb, and run_local.ipynb without
executing any cells. Checks that each notebook contains the required
constants, helper functions, HP-search loop, and platform-specific wiring.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# (notebook_filename, expected_BASELINE, expected_CONFIG_NAME stem)
_NOTEBOOK_SPECS = [
    ("run_kaggle.ipynb", "quartznet_10x4", "quartznet_10x4.yaml"),
    ("run_colab.ipynb", "efficient_conformer", "efficient_conformer.yaml"),
    ("run_local.ipynb", "crdnn", "crdnn.yaml"),
]


def _load_notebook(filename: str) -> nbformat.NotebookNode:
    path = _NOTEBOOKS_DIR / filename
    assert path.exists(), f"Notebook not found: {path}"
    with open(path, encoding="utf-8") as fh:
        return nbformat.read(fh, as_version=4)


def _code_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [cell.get("source", "") for cell in nb.cells if cell.cell_type == "code"]


def _all_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [cell.get("source", "") for cell in nb.cells]


def _combined_code(nb: nbformat.NotebookNode) -> str:
    return "\n".join(_code_sources(nb))


# ---------------------------------------------------------------------------
# Parametrized helpers
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=_NOTEBOOK_SPECS,
    ids=[spec[0] for spec in _NOTEBOOK_SPECS],
)
def nb_spec(request):
    """Yields (notebook_node, baseline, config_name) tuples."""
    filename, baseline, config_name = request.param
    nb = _load_notebook(filename)
    return nb, baseline, config_name, filename


# ---------------------------------------------------------------------------
# Test 1 – notebook parses as valid nbformat-4
# ---------------------------------------------------------------------------


def test_notebook_parses(nb_spec):
    nb, _, _, filename = nb_spec
    assert nb is not None, f"{filename} failed to parse"
    assert len(nb.cells) > 0, f"{filename} has no cells"
    assert nb.nbformat >= 4, f"{filename} must be nbformat >= 4"


# ---------------------------------------------------------------------------
# Test 2 – BASELINE constant
# ---------------------------------------------------------------------------


def test_baseline_constant_defined(nb_spec):
    nb, baseline, _, filename = nb_spec
    code = _combined_code(nb)
    assert f'BASELINE = "{baseline}"' in code, (
        f'{filename}: expected BASELINE = "{baseline}" in a code cell'
    )


# ---------------------------------------------------------------------------
# Test 3 – CONFIG_NAME constant
# ---------------------------------------------------------------------------


def test_config_name_constant_defined(nb_spec):
    nb, _, config_name, filename = nb_spec
    code = _combined_code(nb)
    assert f'CONFIG_NAME = "{config_name}"' in code, (
        f'{filename}: expected CONFIG_NAME = "{config_name}" in a code cell'
    )


# ---------------------------------------------------------------------------
# Test 4 – N_TRIALS constant with integer value
# ---------------------------------------------------------------------------


def test_n_trials_constant_defined(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "N_TRIALS = " in code, (
        f"{filename}: expected N_TRIALS = <int> in a code cell"
    )
    # Verify the line contains a numeric value after the assignment
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("N_TRIALS = "):
            value_str = stripped.split("=", 1)[1].strip()
            assert value_str.isdigit(), (
                f"{filename}: N_TRIALS must be assigned an integer literal, got: {value_str!r}"
            )
            break


# ---------------------------------------------------------------------------
# Test 5 – sample_params function
# ---------------------------------------------------------------------------


def test_sample_params_function_defined(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "def sample_params(" in code, (
        f"{filename}: expected def sample_params( in a code cell"
    )


# ---------------------------------------------------------------------------
# Test 6 – patch_config function
# ---------------------------------------------------------------------------


def test_patch_config_function_defined(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "def patch_config(" in code, (
        f"{filename}: expected def patch_config( in a code cell"
    )


# ---------------------------------------------------------------------------
# Test 7 – HP-search loop
# ---------------------------------------------------------------------------


def test_hp_search_loop_present(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "for trial_id in range(N_TRIALS)" in code, (
        f"{filename}: expected 'for trial_id in range(N_TRIALS)' loop in a code cell"
    )


# ---------------------------------------------------------------------------
# Test 8 – reads result.json per trial
# ---------------------------------------------------------------------------


def test_result_json_reference(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "result.json" in code, (
        f"{filename}: expected a reference to 'result.json' for reading per-trial results"
    )


# ---------------------------------------------------------------------------
# Test 9 – writes submission.csv
# ---------------------------------------------------------------------------


def test_submission_csv_reference(nb_spec):
    nb, _, _, filename = nb_spec
    code = _combined_code(nb)
    assert "submission.csv" in code, (
        f"{filename}: expected a reference to 'submission.csv'"
    )


# ---------------------------------------------------------------------------
# Test 10 – platform-specific wiring
# ---------------------------------------------------------------------------


def test_kaggle_notebook_references_kaggle_working(nb_spec):
    nb, _, _, filename = nb_spec
    if filename != "run_kaggle.ipynb":
        pytest.skip("Platform-specific check only for run_kaggle.ipynb")
    code = _combined_code(nb)
    assert "/kaggle/working" in code, (
        "run_kaggle.ipynb: expected '/kaggle/working' path reference"
    )


def test_colab_notebook_references_content_or_drive(nb_spec):
    nb, _, _, filename = nb_spec
    if filename != "run_colab.ipynb":
        pytest.skip("Platform-specific check only for run_colab.ipynb")
    code = _combined_code(nb)
    has_content = "/content" in code
    has_drive = "drive" in code.lower()
    assert has_content or has_drive, (
        "run_colab.ipynb: expected '/content' or Drive mount reference"
    )


def test_local_notebook_references_data_test(nb_spec):
    nb, _, _, filename = nb_spec
    if filename != "run_local.ipynb":
        pytest.skip("Platform-specific check only for run_local.ipynb")
    all_src = "\n".join(_all_sources(nb))
    assert "data/test" in all_src, (
        "run_local.ipynb: expected 'data/test' path reference"
    )
