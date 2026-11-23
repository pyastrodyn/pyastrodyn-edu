"""Every notebook must run from the first cell to the last.

Skipped when the notebook tools are not installed (``pip install -e ".[dev]"``).
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("matplotlib")
pytest.importorskip("ipykernel")

ROOT = Path(__file__).resolve().parents[1]
# The lecture notebooks and the notebooks of the example problems.
NOTEBOOKS = (sorted((ROOT / "notebooks").glob("*.ipynb"))
             + sorted((ROOT / "examples").glob("lecture*/*.ipynb")))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path):
    notebook = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3").execute()
