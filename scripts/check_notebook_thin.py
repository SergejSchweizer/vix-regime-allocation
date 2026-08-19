"""Fail if the canonical notebook contains implementation code instead of helper calls."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NOTEBOOK = Path("notebooks/gwp2_vix_regime_allocation.ipynb")
IMPORT_CELL = "import vix_regime_allocation.notebook_helpers as nb"
CALL_PATTERN = re.compile(r"nb\.show_[a-z0-9_]+\(\)")


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise SystemExit("Canonical notebook must contain helper-call cells.")

    first = str(code_cells[0].source).strip()
    if first != IMPORT_CELL:
        raise SystemExit(
            "First notebook code cell must contain only the notebook_helpers import alias."
        )

    for position, cell in enumerate(code_cells[1:], start=2):
        source = str(cell.source).strip()
        if not CALL_PATTERN.fullmatch(source):
            raise SystemExit(
                f"Notebook code cell {position} contains implementation code instead of one "
                f"helper call: {source!r}"
            )
        if not cell.outputs:
            raise SystemExit(f"Notebook helper-call cell {position} has no persisted output.")

    for cell in code_cells:
        if cell.get("execution_count") is None:
            raise SystemExit("Every notebook code cell must be executed and persisted.")

    print(
        f"thin notebook contract passed: {len(code_cells) - 1} helper-call cells, "
        "no analysis implementation in the notebook"
    )


if __name__ == "__main__":
    main()
