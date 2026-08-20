from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gwp2_vix_regime_allocation.ipynb"

DISPLAY_DOLLAR_RE = re.compile(r"(?s)\$\$(.*?)\$\$")
DISPLAY_BRACKET_RE = re.compile(r"(?s)\\\[(.*?)\\\]")
INLINE_PAREN_RE = re.compile(r"(?s)\\\((.*?)\\\)")
INLINE_DOLLAR_RE = re.compile(r"(?s)(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")


def _normalize_math_body(body: str) -> str:
    # A non-raw Python string containing ``\neq`` can be serialized as a line break
    # followed by ``eq``. Repair that exact corruption, with or without a literal
    # backslash left immediately before the line break.
    body = re.sub(
        r"(?:\\)?\n[ \t]*eq\b",
        r"\\neq",
        body,
    )
    return body


def _replace_math_spans(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        original = match.group(0)
        body = match.group(1)
        normalized = _normalize_math_body(body)
        if normalized == body:
            return original
        replacements += 1
        return original[: match.start(1) - match.start(0)] + normalized + original[
            match.end(1) - match.start(0) :
        ]

    return pattern.sub(replace, text), replacements


def normalize_markdown_math(text: str) -> tuple[str, int]:
    total = 0
    for pattern in (
        DISPLAY_DOLLAR_RE,
        DISPLAY_BRACKET_RE,
        INLINE_PAREN_RE,
        INLINE_DOLLAR_RE,
    ):
        text, count = _replace_math_spans(text, pattern)
        total += count
    return text, total


def normalize_notebook(path: Path = NOTEBOOK) -> int:
    notebook = nbformat.read(path, as_version=4)
    replacements = 0

    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = str(cell.get("source", ""))
        normalized, count = normalize_markdown_math(source)
        if count:
            cell.source = normalized
            replacements += count

    if replacements:
        nbformat.validate(notebook)
        nbformat.write(notebook, path)

    return replacements


def main() -> None:
    replacements = normalize_notebook()
    print(f"Normalized {replacements} malformed LaTeX math span(s).")


if __name__ == "__main__":
    main()
