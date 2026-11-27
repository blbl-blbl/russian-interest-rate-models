"""Sanity checks for the formulas used in the saved Vasicek research notebook.

These tests do not require local CBR data and do not require notebook outputs
to be empty. Published graphs and tables are intentionally retained.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "02_vasicek_calibration.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_function(name):
    """Extract only the named function definition, without running data-loading cells."""
    for cell in notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        syntax = ast.parse(source)
        for node in syntax.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                module = ast.fix_missing_locations(
                    ast.Module(body=[node], type_ignores=[])
                )
                env = {"np": np}
                exec(compile(module, str(NOTEBOOK), "exec"), env)
                return env[name]
    raise AssertionError(f"Notebook function missing: {name}")


def example_transitions():
    r_prev = np.array([0.070, 0.072, 0.071, 0.076, 0.074, 0.080, 0.077])
    r_next = np.array([0.072, 0.071, 0.076, 0.074, 0.080, 0.077, 0.079])
    delta_t = np.array([1, 3, 1, 1, 2, 3, 1], dtype=float) / 365
    return r_prev, r_next, delta_t


def test_notebook_code_cells_are_valid_python():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_profile_nll_matches_full_likelihood_at_analytical_parameters():
    profile_params = notebook_function("profile_params")
    vasicek_nll = notebook_function("vasicek_nll")
    r0, r1, dt = example_transitions()
    for a in [0.1, 0.8, 3.0]:
        b, sigma, profiled_nll = profile_params(a, r0, r1, dt)
        full_nll = vasicek_nll((a, b, sigma), r0, r1, dt)
        assert np.isfinite(profiled_nll)
        assert sigma > 0
        assert full_nll == pytest.approx(profiled_nll, abs=1e-8)


def test_full_likelihood_penalizes_collapsed_variance():
    vasicek_nll = notebook_function("vasicek_nll")
    r0, r1, dt = example_transitions()
    moderate = vasicek_nll((0.8, 0.08, 0.03), r0, r1, dt)
    nearly_zero = vasicek_nll((0.8, 0.08, 1e-6), r0, r1, dt)
    assert nearly_zero > moderate


def test_notebook_contains_source_and_summary():
    text = "\n".join(
        "".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "markdown"
    )
    assert "Калибровка и диагностика" in text
    assert "Основные итоги" in text
