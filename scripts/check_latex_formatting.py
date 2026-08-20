from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gwp2_vix_regime_allocation.ipynb"

DISPLAY_DOLLAR_RE = re.compile(r"(?s)\$\$(.*?)\$\$")
DISPLAY_BRACKET_RE = re.compile(r"(?s)\\\[(.*?)\\\]")
INLINE_PAREN_RE = re.compile(r"(?s)\\\((.*?)\\\)")
INLINE_DOLLAR_RE = re.compile(r"(?s)(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")
BEGIN_END_RE = re.compile(r"\\(begin|end)\{([^{}]+)\}")
SUSPICIOUS_SPLIT_RE = re.compile(
    r"(?m)(?:\\)?\n[ \t]*(eq|egin|rac|heta|imes|ight|abla|eg)\b"
)


@dataclass(frozen=True)
class MathSpan:
    label: str
    body: str


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _balanced_braces(body: str) -> bool:
    depth = 0
    for index, char in enumerate(body):
        if char not in "{}" or _is_escaped(body, index):
            continue
        if char == "{":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _balanced_environments(body: str) -> bool:
    stack: list[str] = []
    for match in BEGIN_END_RE.finditer(body):
        kind, name = match.groups()
        if kind == "begin":
            stack.append(name)
            continue
        if not stack or stack.pop() != name:
            return False
    return not stack


def _mask_spans(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _math_spans(text: str) -> list[MathSpan]:
    spans: list[MathSpan] = []
    patterns = (
        ("display $$...$$", DISPLAY_DOLLAR_RE),
        ("display \\[...\\]", DISPLAY_BRACKET_RE),
        ("inline \\(...\\)", INLINE_PAREN_RE),
        ("inline $...$", INLINE_DOLLAR_RE),
    )
    for label, pattern in patterns:
        for match in pattern.finditer(text):
            spans.append(MathSpan(label=label, body=match.group(1)))
    return spans


def _delimiter_errors(text: str) -> list[str]:
    errors: list[str] = []
    if text.count("$$") % 2:
        errors.append("unbalanced $$ display-math delimiters")
    if text.count(r"\[") != text.count(r"\]"):
        errors.append("unbalanced \\[...\\] display-math delimiters")
    if text.count(r"\(") != text.count(r"\)"):
        errors.append("unbalanced \\(...\\) inline-math delimiters")

    masked = _mask_spans(text, DISPLAY_DOLLAR_RE)
    masked = _mask_spans(masked, DISPLAY_BRACKET_RE)
    masked = _mask_spans(masked, INLINE_PAREN_RE)
    single_dollars = [
        index
        for index, char in enumerate(masked)
        if char == "$" and not _is_escaped(masked, index)
    ]
    if len(single_dollars) % 2:
        errors.append("unbalanced single-$ inline-math delimiters")
    return errors


def _span_errors(span: MathSpan) -> list[str]:
    errors: list[str] = []
    control_chars = sorted(
        {ord(char) for char in span.body if ord(char) < 32 and char != "\n"}
    )
    if control_chars:
        codes = ", ".join(f"U+{code:04X}" for code in control_chars)
        errors.append(f"contains control character(s) {codes}")
    suspicious = SUSPICIOUS_SPLIT_RE.search(span.body)
    if suspicious:
        errors.append(
            "contains a split LaTeX command around line break "
            f"('{suspicious.group(1)}')"
        )
    if not _balanced_braces(span.body):
        errors.append("has unbalanced LaTeX braces")
    if not _balanced_environments(span.body):
        errors.append("has mismatched \\begin{...}/\\end{...} environments")
    return errors


def validate_notebook_latex(path: Path = NOTEBOOK) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    failures: list[str] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        source = str(cell.get("source", ""))
        for error in _delimiter_errors(source):
            failures.append(f"markdown cell {cell_index}: {error}")
        for span_index, span in enumerate(_math_spans(source), start=1):
            for error in _span_errors(span):
                failures.append(
                    f"markdown cell {cell_index}, {span.label} #{span_index}: {error}"
                )

    return failures


def main() -> None:
    failures = validate_notebook_latex()
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"LaTeX validation failed:\n{details}")
    print("All notebook Markdown formulas passed LaTeX formatting checks.")


if __name__ == "__main__":
    main()
