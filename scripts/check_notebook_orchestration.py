"""Enforce that the GWP2 notebook contains orchestration calls only."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
HELPER = ROOT / "src/vix_regime_allocation/notebook_helpers.py"
ALLOWED_IMPORT = "from vix_regime_allocation import notebook_helpers as nb"


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _helper_functions() -> set[str]:
    tree = ast.parse(HELPER.read_text(encoding="utf-8"), filename=str(HELPER))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _call_name(source: str) -> str | None:
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    expression = tree.body[0].value
    if not isinstance(expression, ast.Call) or expression.args or expression.keywords:
        return None
    function = expression.func
    if not isinstance(function, ast.Attribute):
        return None
    if not isinstance(function.value, ast.Name) or function.value.id != "nb":
        return None
    return function.attr


def validate_orchestration() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    helpers = _helper_functions()
    import_seen = False
    calls: list[str] = []
    violations: list[str] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _source_text(cell).strip()
        if not source:
            continue
        if source == ALLOWED_IMPORT:
            if import_seen:
                violations.append(f"cell {index}: duplicate helper import")
            import_seen = True
            continue
        call_name = _call_name(source)
        if call_name is None:
            violations.append(f"cell {index}: implementation code is not allowed: {source[:120]!r}")
            continue
        if not import_seen:
            violations.append(f"cell {index}: helper call appears before helper import")
        if call_name not in helpers:
            violations.append(f"cell {index}: helper {call_name!r} does not exist")
        calls.append(call_name)

    if not import_seen:
        violations.append("missing notebook_helpers import")
    if not calls:
        violations.append("notebook contains no helper calls")
    duplicate_calls = sorted({name for name in calls if calls.count(name) > 1})
    if duplicate_calls:
        violations.append(f"duplicate helper calls: {duplicate_calls}")

    if violations:
        raise SystemExit("Notebook orchestration contract failed:\n- " + "\n- ".join(violations))

    print(f"Notebook orchestration contract passed: {len(calls)} helper calls, no implementation code.")
    return len(calls)


def main() -> None:
    validate_orchestration()


if __name__ == "__main__":
    main()
