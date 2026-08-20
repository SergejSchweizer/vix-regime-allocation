"""Regression contract for the user-visible Step 5 cumulative comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gwp2_vix_regime_allocation.ipynb"
STEP5_CALL = "nb.step_5_performance_metrics_and_cumulative_compar_034()"


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_step5_cumulative_comparison_is_embedded_and_portable() -> None:
    """The cumulative visual must render inside the notebook on any machine."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matching = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and _source_text(cell).strip() == STEP5_CALL
    ]
    assert len(matching) == 1

    outputs = matching[0].get("outputs", [])
    assert any(
        output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
        for output in outputs
    ), "Step 5 cumulative comparison must be stored as an embedded PNG output."

    serialized = json.dumps(notebook)
    assert "/home/runner/work/" not in serialized
    assert "![Cumulative performance comparison](/" not in serialized
