"""Enforce the helper-only HMM notebook orchestration contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
HELPERS = {
    "nb": ("notebook_helpers", ROOT / "src/vix_regime_allocation/notebook_helpers.py"),
    "sensitivity_nb": (
        "notebook_sensitivity",
        ROOT / "src/vix_regime_allocation/notebook_sensitivity.py",
    ),
}
REQUIRED_CALLS: tuple[tuple[str, str], ...] = (
    ("nb", "step_1_data_overview"),
    ("nb", "step_2_hmm_diagnostics"),
    ("nb", "step_3_hmm_selection"),
    ("nb", "step_4_dual_allocations"),
    ("nb", "step_5_hmm_dual_method_comparison"),
    ("sensitivity_nb", "step_5_state_count_sensitivity"),
    ("nb", "canonical_works_cited"),
)


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _helper_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _import_alias(source: str) -> str | None:
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ImportFrom):
        return None
    statement = tree.body[0]
    if statement.module != "vix_regime_allocation" or len(statement.names) != 1:
        return None
    imported = statement.names[0]
    if imported.asname is None or imported.asname not in HELPERS:
        return None
    expected_module, _ = HELPERS[imported.asname]
    if imported.name != expected_module:
        return None
    return imported.asname


def _call_target(source: str) -> tuple[str, str] | None:
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    expression = tree.body[0].value
    if not isinstance(expression, ast.Call) or expression.args or expression.keywords:
        return None
    function = expression.func
    if not isinstance(function, ast.Attribute):
        return None
    if not isinstance(function.value, ast.Name) or function.value.id not in HELPERS:
        return None
    return function.value.id, function.attr


def _validate_helper_surface(helper_functions: dict[str, set[str]]) -> list[str]:
    violations: list[str] = []
    for alias, call_name in REQUIRED_CALLS:
        if call_name not in helper_functions.get(alias, set()):
            violations.append(f"required helper {alias}.{call_name} does not exist")
    return violations


def _legacy_staging_mode(
    calls: list[tuple[str, str]], implementation_cells: list[tuple[int, str]]
) -> bool:
    """Allow the pre-PR-61 notebook until the complete new call surface is present."""
    if implementation_cells or not calls:
        return False
    return not all(required in calls for required in REQUIRED_CALLS)


def validate_orchestration() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    helper_functions = {
        alias: _helper_functions(path) for alias, (_, path) in HELPERS.items() if path.is_file()
    }
    imported_aliases: set[str] = set()
    import_positions: dict[str, int] = {}
    call_records: list[tuple[int, tuple[str, str]]] = []
    implementation_cells: list[tuple[int, str]] = []
    violations = _validate_helper_surface(helper_functions)

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _source_text(cell).strip()
        if not source:
            continue

        alias = _import_alias(source)
        if alias is not None:
            if alias in imported_aliases:
                violations.append(f"cell {index}: duplicate helper import for {alias!r}")
            imported_aliases.add(alias)
            import_positions.setdefault(alias, index)
            continue

        target = _call_target(source)
        if target is None:
            implementation_cells.append((index, source))
            continue
        call_records.append((index, target))

    calls = [target for _, target in call_records]
    if _legacy_staging_mode(calls, implementation_cells):
        if violations:
            raise SystemExit("Notebook helper surface failed:\n- " + "\n- ".join(violations))
        print(
            "Notebook orchestration staging contract passed: the HMM-only helper surface is "
            "complete; strict notebook validation activates when PR-61 adopts every new call."
        )
        return len(calls)

    for index, source in implementation_cells:
        violations.append(f"cell {index}: implementation code is not allowed: {source[:120]!r}")

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _source_text(cell).strip()
        if source and "markov" in source.lower():
            violations.append(f"cell {index}: forbidden non-HMM helper/code reference")

    for alias in HELPERS:
        if alias not in imported_aliases:
            violations.append(f"missing {alias} helper import")

    duplicate_calls = sorted({target for target in calls if calls.count(target) > 1})
    if duplicate_calls:
        violations.append(f"duplicate helper calls: {duplicate_calls}")

    for index, target in call_records:
        alias, call_name = target
        import_index = import_positions.get(alias)
        if import_index is None or index < import_index:
            violations.append(f"cell {index}: helper call appears before import for {alias!r}")
        if call_name not in helper_functions.get(alias, set()):
            violations.append(f"cell {index}: helper {alias}.{call_name} does not exist")

    for required in REQUIRED_CALLS:
        count = calls.count(required)
        if count != 1:
            message = f"required helper call {required[0]}.{required[1]} occurs {count} times"
            violations.append(message)

    unexpected = sorted(set(calls) - set(REQUIRED_CALLS))
    if unexpected:
        violations.append(f"unexpected analytical helper calls: {unexpected}")

    if violations:
        raise SystemExit("Notebook orchestration contract failed:\n- " + "\n- ".join(violations))

    print(
        f"Notebook orchestration contract passed: {len(calls)} HMM-only helper calls, "
        "no embedded analytical code."
    )
    return len(calls)


def main() -> None:
    validate_orchestration()


if __name__ == "__main__":
    main()
