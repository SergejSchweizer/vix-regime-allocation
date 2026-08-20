from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

# Programmatically generated Markdown can corrupt LaTeX when a normal Python
# string contains a single backslash.  Python consumes \a, \b, \f, \n, \r,
# \t, and \v as control escapes before nbformat ever sees the text.  Repair
# those artifacts only inside math delimiters so prose and code stay untouched.
MATH_SPAN = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)",
    re.DOTALL,
)

# These control characters are not meaningful formatting inside a LaTeX math
# span.  Reconstruct the command prefix and preserve the following letters, so
# examples such as backspace + "egin" and form-feed + "rac" become \begin and
# \frac without hard-coding every b/f/r/a/v-prefixed command.
CONTROL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("\a", "a"),
    ("\b", "b"),
    ("\f", "f"),
    ("\r", "r"),
    ("\v", "v"),
)

# Newline and tab are legitimate layout characters, so repair only known LaTeX
# commands whose leading \n or \t may have been consumed by Python.
NEWLINE_COMMANDS = (
    "nabla",
    "neq",
    "nonumber",
    "not",
    "nu",
)
TAB_COMMANDS = (
    "tau",
    "text",
    "tfrac",
    "theta",
    "therefore",
    "tilde",
    "times",
    "to",
    "top",
)


def _repair_control_prefixes(text: str) -> str:
    for control, prefix in CONTROL_PREFIXES:
        pattern = re.compile(re.escape(control) + r"([A-Za-z]+)")
        text = pattern.sub(
            lambda match, prefix=prefix: "\\" + prefix + match.group(1),
            text,
        )
    return text


def _repair_known_layout_escapes(text: str) -> str:
    for command in NEWLINE_COMMANDS:
        text = text.replace("\n" + command[1:], "\\" + command)
    for command in TAB_COMMANDS:
        text = text.replace("\t" + command[1:], "\\" + command)
    return text


def _repair_math(match: re.Match[str]) -> str:
    text = _repair_control_prefixes(match.group(0))
    return _repair_known_layout_escapes(text)


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
