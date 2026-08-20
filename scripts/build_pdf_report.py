from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gwp2_vix_regime_allocation.ipynb"
TEMPLATE = ROOT / "reports" / "Template_Stochastic_Modeling_Group_Work_Project.pdf"
OUTPUT = ROOT / "reports" / "Stochastic_Modeling_GWP2_Report.pdf"
MATH_RENDERING_VERSION = "mathjax-latex-v1"

GROUP_NUMBER = "739"
GROUP_MEMBERS = (
    "Umuhoza Denyse Graine",
    "Opeyemi Waliyilah Oladipupo",
    "Sergej Schweizer",
)

PRINT_MATH_STYLE = """
<style id="report-math-print-style">
@media print {
  mjx-container[display="true"],
  .MathJax_Display {
    break-inside: avoid;
    page-break-inside: avoid;
    overflow: visible !important;
  }
}
</style>
"""

MATH_READY_SCRIPT = """
<script id="report-math-ready-script">
window.addEventListener("load", async () => {
  try {
    if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
      await window.MathJax.startup.promise;
    }
    if (window.MathJax && window.MathJax.typesetPromise) {
      await window.MathJax.typesetPromise();
    }
    document.documentElement.dataset.mathjaxReady = "true";
  } catch (error) {
    document.documentElement.dataset.mathjaxReady = "error";
    console.error("MathJax rendering failed", error);
  }
});
</script>
"""


def _notebook_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    failures: list[str] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append(
                    f"cell {index}: {output.get('ename', 'Error')}: {output.get('evalue', '')}"
                )
    if failures:
        raise RuntimeError("Notebook contains failed outputs: " + "; ".join(failures))


def _export_notebook_html(notebook_path: Path, html_path: Path) -> None:
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = False
    exporter.exclude_output = False
    body, _ = exporter.from_filename(str(notebook_path))
    if "</head>" not in body:
        raise RuntimeError("nbconvert HTML export is missing a closing head element.")
    body = body.replace(
        "</head>",
        f"{PRINT_MATH_STYLE}{MATH_READY_SCRIPT}</head>",
        1,
    )
    html_path.write_text(body, encoding="utf-8")


def _find_chrome() -> str:
    configured = os.environ.get("CHROME_BIN")
    if configured:
        return configured

    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise RuntimeError(
        "A Chromium-based browser is required. Set CHROME_BIN or install Google Chrome/Chromium."
    )


def _print_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = _find_chrome()
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("Chrome did not produce a non-empty notebook PDF.")


def _template_cover(template_path: Path) -> object:
    reader = PdfReader(str(template_path))
    if not reader.pages:
        raise RuntimeError("The supplied PDF template has no pages.")

    cover = reader.pages[0]
    width = float(cover.mediabox.width)
    height = float(cover.mediabox.height)

    overlay_bytes = BytesIO()
    overlay = canvas.Canvas(overlay_bytes, pagesize=(width, height))

    # Keep the supplied template title/course framing and replace only the lower
    # group-information area so the report carries the actual team membership.
    lower_area_height = height * 0.40
    overlay.setFillColorRGB(1, 1, 1)
    overlay.rect(0, 0, width, lower_area_height, stroke=0, fill=1)

    left = width * 0.16
    y = height * 0.33
    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont("Helvetica-Bold", 12)
    overlay.drawString(left, y, f"Group #: {GROUP_NUMBER}")
    y -= 30
    overlay.drawString(left, y, "Group Members:")
    overlay.setFont("Helvetica", 11)
    for member in GROUP_MEMBERS:
        y -= 22
        overlay.drawString(left + 18, y, member)

    overlay.save()
    overlay_bytes.seek(0)
    cover.merge_page(PdfReader(overlay_bytes).pages[0])
    return cover


def build_report(
    notebook_path: Path = NOTEBOOK,
    template_path: Path = TEMPLATE,
    output_path: Path = OUTPUT,
) -> Path:
    """Build the PDF strictly as a derived sidecar of the executed notebook.

    The notebook is the single source of truth. The PDF adds only the supplied course cover
    and then renders the notebook in order; it must never introduce independent analysis,
    equations, explanations, tables, or figures.
    """
    _validate_notebook(notebook_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gwp2-report-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        html_path = temp_dir / "notebook.html"
        notebook_pdf = temp_dir / "notebook.pdf"
        _export_notebook_html(notebook_path, html_path)
        _print_html_to_pdf(html_path, notebook_pdf)

        writer = PdfWriter()
        writer.add_page(_template_cover(template_path))
        body = PdfReader(str(notebook_pdf))
        for page in body.pages:
            writer.add_page(page)

        writer.add_metadata(
            {
                "/Title": "MScFE 622 Group Work Project 2 — VIX Regime Allocation",
                "/Subject": "Derived PDF sidecar of the canonical executed notebook",
                "/ArtifactRole": "notebook-sidecar",
                "/SourceOfTruth": "notebooks/gwp2_vix_regime_allocation.ipynb",
                "/NotebookSHA256": _notebook_sha256(notebook_path),
                "/MathRenderingVersion": MATH_RENDERING_VERSION,
            }
        )
        with output_path.open("wb") as handle:
            writer.write(handle)

    if output_path.stat().st_size == 0:
        raise RuntimeError("Generated report is empty.")
    return output_path


def main() -> None:
    report = build_report()
    print(f"Generated {report.relative_to(ROOT)} ({report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
