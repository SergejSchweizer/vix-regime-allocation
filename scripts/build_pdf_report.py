from __future__ import annotations

import hashlib
import os
import re
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

GROUP_NUMBER = "739"
GROUP_MEMBERS = (
    "Umuhoza Denyse Graine",
    "Opeyemi Waliyilah Oladipupo",
    "Sergej Schweizer",
)
MATHJAX_TIMEOUT_MS = 30_000

MATHJAX_READINESS_SCRIPT = r"""
<style id="pdf-math-layout">
  mjx-container,
  .MathJax,
  .MathJax_Display,
  .MathJax_SVG {
    break-inside: avoid;
    page-break-inside: avoid;
  }
</style>
<script id="pdf-mathjax-readiness">
(() => {
  const root = document.documentElement;
  const deadline = Date.now() + 20000;
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const renderedMathCount = () => document.querySelectorAll(
    "mjx-container, .MathJax, .MathJax_Display, .MathJax_SVG"
  ).length;

  const mark = (state, detail = "") => {
    root.dataset.mathReady = state;
    root.dataset.mathRenderCount = String(renderedMathCount());
    if (detail) {
      root.dataset.mathReadyDetail = detail.slice(0, 300);
    }
  };

  const finishReady = () => {
    const count = renderedMathCount();
    if (count <= 0) {
      mark("failed", "MathJax reported ready but no rendered math nodes were found.");
      return;
    }
    mark("ready");
  };

  async function waitForMathJax() {
    while (Date.now() < deadline) {
      const mathJax = window.MathJax;
      if (mathJax && mathJax.startup && mathJax.startup.promise) {
        try {
          await mathJax.startup.promise;
          finishReady();
        } catch (error) {
          mark("failed", String(error));
        }
        return;
      }
      if (mathJax && mathJax.Hub && typeof mathJax.Hub.Queue === "function") {
        try {
          mathJax.Hub.Queue(["Typeset", mathJax.Hub]);
          mathJax.Hub.Queue(() => finishReady());
        } catch (error) {
          mark("failed", String(error));
        }
        return;
      }
      await sleep(100);
    }
    mark("failed", "MathJax did not expose a supported API before the timeout.");
  }

  const start = () => {
    root.dataset.mathReady = "waiting";
    void waitForMathJax();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
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


def _inject_mathjax_readiness(html: str) -> str:
    if "</body>" in html:
        return html.replace("</body>", MATHJAX_READINESS_SCRIPT + "\n</body>", 1)
    return html + MATHJAX_READINESS_SCRIPT


def _export_notebook_html(notebook_path: Path, html_path: Path) -> None:
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = False
    exporter.exclude_output = False
    body, _ = exporter.from_filename(str(notebook_path))
    html_path.write_text(_inject_mathjax_readiness(body), encoding="utf-8")


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


def _chrome_base_command(chrome: str) -> list[str]:
    return [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-background-timer-throttling",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        f"--virtual-time-budget={MATHJAX_TIMEOUT_MS}",
    ]


def _verify_mathjax_rendering(chrome: str, html_path: Path) -> None:
    command = [*_chrome_base_command(chrome), "--dump-dom", html_path.resolve().as_uri()]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    state_match = re.search(r'data-math-ready="([^"]+)"', result.stdout)
    count_match = re.search(r'data-math-render-count="(\d+)"', result.stdout)
    state = state_match.group(1) if state_match else "missing"
    count = int(count_match.group(1)) if count_match else 0
    if state != "ready" or count <= 0:
        detail_match = re.search(r'data-math-ready-detail="([^"]*)"', result.stdout)
        detail = detail_match.group(1) if detail_match else "no readiness detail"
        raise RuntimeError(
            "MathJax did not finish rendering notebook formulas before PDF generation: "
            f"state={state!r}, rendered_nodes={count}, detail={detail!r}."
        )


def _print_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = _find_chrome()
    _verify_mathjax_rendering(chrome, html_path)
    command = [
        *_chrome_base_command(chrome),
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
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
