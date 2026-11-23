"""The example problems: every script runs, and every notebook matches its script.

Each example problem is a script ``examples/lectureNN/example_*.py`` and a
notebook of the same name made from it by ``tools/make_examples.py``.
"""

import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "examples").glob("lecture*/example_*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_runs(path, monkeypatch):
    """The script runs to its end; its own ``assert`` lines check the results."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")            # draw figures without opening a window
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        plt.close("all")                               # do not keep the figures of all the scripts open


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_notebook_matches_script(path):
    """The stored notebook is the one that ``tools/make_examples.py`` makes now."""
    pytest.importorskip("nbformat")
    spec = importlib.util.spec_from_file_location("make_examples", ROOT / "tools" / "make_examples.py")
    make_examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_examples)
    import nbformat
    notebook = path.with_suffix(".ipynb")
    assert notebook.exists(), "run: python tools/make_examples.py"
    stored = [(cell.cell_type, cell.source) for cell in nbformat.read(notebook, as_version=4).cells]
    assert stored == make_examples.cells(path.read_text()), "run: python tools/make_examples.py"


def test_every_problem_has_an_example():
    """One script per validation problem, named after the number of the problem."""
    from pyastrodyn_edu import validation
    for problem in validation.PROBLEMS:
        number_in_name = problem.id[1:].replace("-", "_")          # "L6-6" -> "6_6"
        found = [p for p in SCRIPTS if p.name.startswith(f"example_{number_in_name}_")]
        assert len(found) == 1, f"{problem.id}: {len(found)} example scripts"
        assert found[0].parent.name == f"lecture{problem.lecture:02d}"
        assert f'validation.compare("{problem.id}"' in found[0].read_text(), problem.id


def test_compare_with_the_stored_results(capsys):
    """``validation.compare``: agreement, changed data, and a wrong result."""
    from pyastrodyn_edu import validation
    problem = validation.get("L4-1")
    data, results = problem.values(), problem.results()
    assert validation.compare("L4-1", data, results) is True
    assert validation.compare("L4-1", {**data, "t": 3600.0}, results) is False
    assert "data were changed" in capsys.readouterr().out
    assert validation.compare("L4-1", data, {**results, "f": 100.0}) is False
    assert "f" in capsys.readouterr().out
