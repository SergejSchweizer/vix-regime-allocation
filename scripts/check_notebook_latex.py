from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import nbformat

MATH_COMMAND = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|log|ln|exp|"
    r"sin|cos|tan|operatorname|mathrm|mathbf|mathbb|mathcal|left|right|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|"
    r"kappa|lambda|mu|nu|xi|pi|rho|varrho|sigma|varsigma|tau|upsilon|phi|"
    r"varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|"
    r"Psi|Omega)\b"
)
INLINE_CODE = re.compile(r"(?<!`)`[^`\n]*`(?!`)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RELATION = re.compile(r"(?:=|≈|≤|≥|<|>)")


@dataclass(frozen=True)
class Problem:
    cell: int
    line: int
    message: str
    excerpt: str

    def render(self) -> str:
        excerpt = self.excerpt.strip().replace("\t", " ")
        return f"cell {self.cell}, line {self.line}: {self.message}: {excerpt}"


def _without_fenced_code(source: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in source.splitlines(keepends=True):
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            kept.append("\n" if line.endswith("\n") else "")
            continue
        if in_fence:
            kept.append("\n" if line.endswith("\n") else "")
        else:
            kept.append(line)
    return "".join(kept)


def _mask_preserving_newlines(match: re.Match[str]) -> str:
    return "".join("\n" if char == "\n" else " " for char in match.group(0))


def _unescaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _matching_single_dollar(text: str, start: int) -> int | None:
    cursor = start + 1
    while cursor < len(text):
        if text[cursor] == "\n":
            return None
        if text[cursor] == "$" and _unescaped(text, cursor):
            if cursor + 1 >= len(text) or text[cursor + 1] != "$":
                return cursor
        cursor += 1
    return None


def _balanced_braces(math_text: str) -> bool:
    depth = 0
    for index, char in enumerate(math_text):
        if not _unescaped(math_text, index):
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _line_excerpt(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return text[start:end]


def _problem(cell: int, text: str, index: int, message: str) -> Problem:
    return Problem(cell, _line_number(text, index), message, _line_excerpt(text, index))


def _scan_math(text: str, cell_index: int) -> tuple[list[Problem], list[tuple[int, str]]]:
    problems: list[Problem] = []
    outside_chunks: list[tuple[int, str]] = []
    index = 0
    outside_start = 0

    def flush_outside(end: int) -> None:
        nonlocal outside_start
        if end > outside_start:
            outside_chunks.append((outside_start, text[outside_start:end]))

    while index < len(text):
        if text.startswith("$$", index) and _unescaped(text, index):
            flush_outside(index)
            close = text.find("$$", index + 2)
            if close == -1:
                problems.append(
                    _problem(cell_index, text, index, "unclosed display-math delimiter '$$'")
                )
                outside_start = len(text)
                break
            math_text = text[index + 2 : close]
            if not math_text.strip():
                problems.append(_problem(cell_index, text, index, "empty display-math block"))
            elif not _balanced_braces(math_text):
                problems.append(
                    _problem(cell_index, text, index, "unbalanced LaTeX braces in display math")
                )
            index = close + 2
            outside_start = index
            continue

        if text.startswith(r"\[", index) and _unescaped(text, index):
            flush_outside(index)
            close = text.find(r"\]", index + 2)
            if close == -1:
                problems.append(
                    _problem(cell_index, text, index, r"unclosed display-math delimiter '\['")
                )
                outside_start = len(text)
                break
            math_text = text[index + 2 : close]
            if not _balanced_braces(math_text):
                problems.append(
                    _problem(cell_index, text, index, "unbalanced LaTeX braces in display math")
                )
            index = close + 2
            outside_start = index
            continue

        if text.startswith(r"\(", index) and _unescaped(text, index):
            flush_outside(index)
            close = text.find(r"\)", index + 2)
            if close == -1:
                problems.append(
                    _problem(cell_index, text, index, r"unclosed inline-math delimiter '\('")
                )
                outside_start = len(text)
                break
            math_text = text[index + 2 : close]
            if not _balanced_braces(math_text):
                problems.append(
                    _problem(cell_index, text, index, "unbalanced LaTeX braces in inline math")
                )
            index = close + 2
            outside_start = index
            continue

        if text[index] == "$" and _unescaped(text, index):
            close = _matching_single_dollar(text, index)
            if close is not None:
                flush_outside(index)
                math_text = text[index + 1 : close]
                if not math_text.strip():
                    problems.append(_problem(cell_index, text, index, "empty inline-math span"))
                elif not _balanced_braces(math_text):
                    problems.append(
                        _problem(cell_index, text, index, "unbalanced LaTeX braces in inline math")
                    )
                index = close + 1
                outside_start = index
                continue
        index += 1

    flush_outside(len(text))
    return problems, outside_chunks


def _scan_cell(source: str, cell_index: int) -> list[Problem]:
    stripped_code = _without_fenced_code(source)
    text = HTML_COMMENT.sub(_mask_preserving_newlines, stripped_code)
    text = INLINE_CODE.sub(lambda match: " " * len(match.group(0)), text)
    problems, outside_chunks = _scan_math(text, cell_index)

    for chunk_start, chunk in outside_chunks:
        for match in MATH_COMMAND.finditer(chunk):
            absolute = chunk_start + match.start()
            problems.append(
                Problem(
                    cell_index,
                    _line_number(text, absolute),
                    f"raw LaTeX command {match.group(0)!r} is outside a math delimiter",
                    _line_excerpt(text, absolute),
                )
            )

        offset = 0
        for line in chunk.splitlines(keepends=True):
            formula = line.strip()
            if formula and RELATION.search(formula):
                word_count = len(re.findall(r"\b[A-Za-z]{3,}\b", formula))
                markdown_table = formula.startswith("|") and formula.endswith("|")
                if not markdown_table and word_count <= 3:
                    absolute = chunk_start + offset + len(line) - len(line.lstrip())
                    problems.append(
                        Problem(
                            cell_index,
                            _line_number(text, absolute),
                            "formula-like text is outside a LaTeX math delimiter",
                            formula,
                        )
                    )
            offset += len(line)

    deduped: list[Problem] = []
    seen: set[tuple[int, int, str]] = set()
    for problem in problems:
        key = (problem.cell, problem.line, problem.message)
        if key not in seen:
            seen.add(key)
            deduped.append(problem)
    return deduped


def check_notebook(path: Path) -> list[Problem]:
    notebook = nbformat.read(path, as_version=4)
    problems: list[Problem] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        problems.extend(_scan_cell(str(source), index))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LaTeX/MathJax formatting in notebook Markdown cells."
    )
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    problems = check_notebook(args.notebook)
    if problems:
        rendered = "\n".join(f"- {problem.render()}" for problem in problems)
        message = (
            f"Notebook LaTeX validation failed with {len(problems)} issue(s):\n{rendered}"
        )
        raise SystemExit(message)
    print(f"Notebook LaTeX validation passed: {args.notebook}")


if __name__ == "__main__":
    main()
