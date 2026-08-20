from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

# Python string literals can silently turn LaTeX commands such as \neq, \theta,
# \frac, \beta, \rho, and \alpha into control characters when Markdown is
# generated programmatically with a single backslash.  Repair only inside math
# delimiters so prose and code remain untouched.
ACCIDENTAL_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\neq", r"\neq"),
    ("\nabla", r"\nabla"),
    ("\nu", r"\nu"),
    ("\theta", r"\theta"),
    ("\tau", r"\tau"),
    ("\text", r"\text"),
    ("\times", r"\times"),
    ("\frac", r"\frac"),
    ("\beta", r"\beta"),
    ("\rho", r"\rho"),
    ("\alpha", r"\alpha"),
)

MATH_SPAN = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)",
    re.DOTALL,
)


def _repair_math(match: re.Match[str]) -> str:
    text = match.group(0)
    for broken, latex in ACCIDENTAL_ESCAPES:
        text = text.replace(broken, latex)
    return text


def normalize_notebook(path: Path) -> bool:
    notebook = nbformat.read(path, as_version=4)
    changed = False

    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = str(cell.get("source", ""))
        repaired = MATH_SPAN.sub(_repair_math, source)
        if repaired != source:
            cell["source"] = repaired
            changed = True

    if changed:
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair accidental Python escape sequences inside notebook LaTeX math."
    )
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    changed = normalize_notebook(args.notebook)
    status = "normalized" if changed else "already normalized"
    print(f"Notebook LaTeX is {status}: {args.notebook}")


if __name__ == "__main__":
    main()
