from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

RAW_TEX = re.compile(
    r"(?:\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|log|ln|exp|"
    r"operatorname|mathrm|mathbf|mathbb|mathcal|left|right|alpha|beta|gamma|"
    r"delta|epsilon|theta|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)\b|\\begin\{|"
    r"\\end\{|\\\[|\\\]|\\\(|\\\)|\$\$)"
)
PAGES_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)


def _require(executable: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        raise RuntimeError(f"Required executable not found: {executable}")
    return resolved


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def validate_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"PDF is missing or empty: {path}")

    pdfinfo = _require("pdfinfo")
    pdftotext = _require("pdftotext")
    pdftoppm = _require("pdftoppm")

    info = _run([pdfinfo, str(path)]).stdout
    match = PAGES_RE.search(info)
    if not match:
        raise RuntimeError(f"Could not read page count from pdfinfo for {path}")
    pages = int(match.group(1))
    if pages < 1:
        raise RuntimeError(f"PDF has no pages: {path}")

    with tempfile.TemporaryDirectory(prefix="latex-pdf-check-") as temp_name:
        temp = Path(temp_name)
        text_path = temp / "report.txt"
        _run([pdftotext, "-layout", "-enc", "UTF-8", str(path), str(text_path)])
        text = text_path.read_text(encoding="utf-8", errors="replace")

        leaked: list[str] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if RAW_TEX.search(line):
                leaked.append(f"line {number}: {line.strip()}")
        if leaked:
            preview = "\n".join(f"- {line}" for line in leaked[:20])
            suffix = "\n- ..." if len(leaked) > 20 else ""
            raise RuntimeError(
                f"Unrendered LaTeX leaked into {path} ({len(leaked)} occurrence(s)):\n"
                f"{preview}{suffix}"
            )

        representative_pages = {1, max(1, (pages + 1) // 2), pages}
        for page in sorted(representative_pages):
            output_prefix = temp / f"page-{page}"
            _run(
                [
                    pdftoppm,
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-singlefile",
                    "-png",
                    "-r",
                    "72",
                    str(path),
                    str(output_prefix),
                ]
            )
            rendered = output_prefix.with_suffix(".png")
            if not rendered.is_file() or rendered.stat().st_size == 0:
                raise RuntimeError(
                    f"Failed to rasterize representative page {page} of {path}"
                )

    print(f"PDF LaTeX/render validation passed: {path} ({pages} pages)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that report PDFs contain rendered math, not raw TeX."
    )
    parser.add_argument("pdf", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.pdf:
        validate_pdf(path)


if __name__ == "__main__":
    main()
