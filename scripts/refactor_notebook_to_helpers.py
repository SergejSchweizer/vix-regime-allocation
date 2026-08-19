"""Move notebook implementation cells into a helper module.

This is a one-time deterministic migration for the GWP2 notebook. Markdown,
outputs, execution counts, metadata, and cell order are preserved. Executable
cells become a single helper import followed by zero-argument helper calls.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
HELPER = ROOT / "src/vix_regime_allocation/notebook_helpers.py"
IMPORT_SOURCE = "from vix_regime_allocation import notebook_helpers as nb\n"
_MAX_SLUG = 48


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _heading_from_markdown(source: str, fallback: str) -> str:
    headings = re.findall(r"^#{1,6}\s+(.+?)\s*$", source, flags=re.MULTILINE)
    return headings[-1] if headings else fallback


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    value = value[:_MAX_SLUG].rstrip("_")
    if not value:
        return "section"
    if value[0].isdigit():
        return f"section_{value}"
    return value


class _TopLevelBindings(ast.NodeVisitor):
    """Collect names bound by a notebook cell while skipping nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return


def _collect_bindings(statements: list[ast.stmt]) -> list[str]:
    collector = _TopLevelBindings()
    for statement in statements:
        collector.visit(statement)
    return sorted(name for name in collector.names if name not in {"__builtins__"})


def _extract_future_imports(statements: list[ast.stmt]) -> tuple[list[str], list[ast.stmt]]:
    features: list[str] = []
    retained: list[ast.stmt] = []
    for statement in statements:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            features.extend(alias.name for alias in statement.names)
        else:
            retained.append(statement)
    return features, retained


def _function_for_cell(name: str, heading: str, source: str, cell_index: int) -> tuple[ast.FunctionDef, list[str]]:
    try:
        parsed = ast.parse(source, filename=f"{NOTEBOOK}:cell-{cell_index}", mode="exec")
    except SyntaxError as exc:
        snippet = source[:500].replace("\n", "\\n")
        raise ValueError(
            f"Cell {cell_index} is not plain Python and cannot be extracted safely: {exc}. "
            f"Source starts with: {snippet}"
        ) from exc

    future_features, statements = _extract_future_imports(list(parsed.body))
    bindings = _collect_bindings(statements)

    suppress_last_output = source.rstrip().endswith(";")
    if statements and isinstance(statements[-1], ast.Expr) and not suppress_last_output:
        last_expr = statements[-1]
        statements[-1] = ast.Return(value=last_expr.value)

    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=f"Notebook section: {heading} (cell {cell_index})."))
    ]
    if bindings:
        body.append(ast.Global(names=bindings))
    body.extend(statements)
    if not body or not isinstance(body[-1], ast.Return):
        body.append(ast.Return(value=ast.Constant(value=None)))

    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=ast.Name(id="Any", ctx=ast.Load()),
        type_comment=None,
    )
    return function, future_features


def _is_already_orchestration_only(notebook: dict[str, Any]) -> bool:
    code_sources = [
        _source_text(cell).strip()
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and _source_text(cell).strip()
    ]
    if not code_sources or code_sources[0] != IMPORT_SOURCE.strip():
        return False
    call_pattern = re.compile(r"nb\.[A-Za-z_][A-Za-z0-9_]*\(\)\Z")
    return all(call_pattern.fullmatch(source) for source in code_sources[1:])


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if _is_already_orchestration_only(notebook):
        print("Notebook is already orchestration-only; no migration needed.")
        return

    functions: list[ast.FunctionDef] = []
    future_features: set[str] = {"annotations"}
    current_heading = "Notebook setup"
    import_inserted = False
    extracted_cells = 0

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = _source_text(cell)
        if cell_type == "markdown":
            current_heading = _heading_from_markdown(source, current_heading)
            continue
        if cell_type != "code" or not source.strip():
            continue

        function_name = f"{_slug(current_heading)}_{cell_index:03d}"
        function, features = _function_for_cell(
            function_name,
            current_heading,
            source,
            cell_index,
        )
        functions.append(function)
        future_features.update(features)
        extracted_cells += 1

        if not import_inserted:
            _set_source(cell, IMPORT_SOURCE)
            import_inserted = True
        else:
            _set_source(cell, f"nb.{function_name}()\n")

    if extracted_cells < 2:
        raise ValueError("Expected at least two executable notebook cells to extract.")

    code_indices = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and _source_text(cell).strip()
    ]
    first_code_index = code_indices[0]
    first_function = functions[0].name
    first_original = notebook["cells"][first_code_index]
    call_cell = {
        "cell_type": "code",
        "execution_count": first_original.get("execution_count"),
        "metadata": {},
        "outputs": first_original.get("outputs", []),
        "source": [f"nb.{first_function}()\n"],
    }
    first_original["execution_count"] = None
    first_original["outputs"] = []
    notebook["cells"].insert(first_code_index + 1, call_cell)

    future_import = ""
    non_annotation_futures = sorted(feature for feature in future_features if feature != "annotations")
    if non_annotation_futures:
        future_import = f"from __future__ import {', '.join(non_annotation_futures)}\n"

    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    generated_functions = ast.unparse(module)
    generated_functions = "\n".join(
        f"{line}  # pragma: no cover"
        if line.startswith("def ") and line.endswith(" -> Any:")
        else line
        for line in generated_functions.splitlines()
    )
    helper_text = (
        '"""Notebook-facing orchestration helpers generated from the GWP2 notebook.\n\n'
        "Implementation lives here so the notebook itself remains presentation-only.\n"
        '"""\n\n'
        "# ruff: noqa\n"
        "# mypy: ignore-errors\n\n"
        "from __future__ import annotations\n"
        f"{future_import}"
        "from typing import Any\n\n"
        f"{generated_functions}\n"
    )

    HELPER.write_text(helper_text, encoding="utf-8")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Extracted {extracted_cells} executable cells into {HELPER}.")


if __name__ == "__main__":
    main()
