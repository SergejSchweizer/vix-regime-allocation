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
import pymupdf
from nbconvert import HTMLExporter
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gwp2_vix_regime_allocation.ipynb"
TEMPLATE = ROOT / "reports" / "Template_Stochastic_Modeling_Group_Work_Project.pdf"
OUTPUT = ROOT / "reports" / "Stochastic_Modeling_GWP2_Report.pdf"
BIBLIOGRAPHY = ROOT / "reports" / "references.bib"

GROUP_WORK_PROJECT_NUMBER = "2"
GROUP_NUMBER = "16855"
REMOVED_MEMBER = "Umuhoza Denyse Graine"
REMAINING_MEMBERS = ("Opeyemi Waliyilah Oladipupo", "Sergej Schweizer")

REQUIRED_CODE_CALLS = (
    "nb.step_1_data_overview()",
    "nb.step_2_hmm_diagnostics()",
    "nb.step_3_hmm_selection()",
    "nb.step_4_dual_allocations()",
    "nb.step_5_hmm_dual_method_comparison()",
    "sensitivity_nb.step_5_state_count_sensitivity()",
    "nb.canonical_works_cited()",
)
REQUIRED_CONTENT_TOKENS = (
    "Hidden Markov Model",
    "Expectation-Maximization",
    "Baum-Welch",
    "100% Keep",
    "60/40 Spread",
    "hmm_100_keep",
    "hmm_60_40_spread",
    "equal_weight_monthly",
    "spy_buy_hold",
    "Works Cited",
)
CITATION_TOKENS = ("Baum", "Rabiner", "Viterbi", "Akaike", "Schwarz")
FORBIDDEN_MARKOV_RESULT_PATTERNS = (
    re.compile(r"\bmarkov\s+k\s*=\s*[23]\b", re.IGNORECASE),
    re.compile(r"step2_markov_", re.IGNORECASE),
    re.compile(r"step4_allocation_mapping\.csv", re.IGNORECASE),
    re.compile(r"markov_vix_states", re.IGNORECASE),
    re.compile(r"\bselected\s+(?:model|family)\s*[:=]\s*markov\b", re.IGNORECASE),
)


def _notebook_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_source(cell: object) -> str:
    source = getattr(cell, "source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return "" if value is None else str(value)


def _output_text(output: object) -> str:
    if not hasattr(output, "get"):
        return ""
    output_type = output.get("output_type")
    if output_type == "stream":
        return _text(output.get("text"))
    if output_type not in {"execute_result", "display_data"}:
        return ""
    data = output.get("data", {})
    if not hasattr(data, "get"):
        return ""
    return "\n".join(
        _text(data.get(mime))
        for mime in ("text/plain", "text/markdown", "text/html")
        if data.get(mime) is not None
    )


def _validate_notebook(path: Path) -> str:
    """Validate the canonical source-of-truth notebook before any PDF rendering."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    failures: list[str] = []
    code_sources: list[str] = []
    content_parts: list[str] = []

    for index, cell in enumerate(notebook.cells):
        source = _cell_source(cell)
        content_parts.append(source)
        if cell.cell_type != "code":
            continue
        if not source.strip():
            continue
        code_sources.append(source.strip())
        if cell.get("execution_count") is None:
            failures.append(f"cell {index}: non-empty code cell is not executed")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append(
                    f"cell {index}: {output.get('ename', 'Error')}: {output.get('evalue', '')}"
                )
            rendered = _output_text(output)
            if rendered:
                content_parts.append(rendered)

    content = "\n".join(content_parts)
    for required in REQUIRED_CODE_CALLS:
        if code_sources.count(required) != 1:
            failures.append(f"required notebook helper call must occur exactly once: {required}")

    for token in REQUIRED_CONTENT_TOKENS:
        if token not in content:
            failures.append(f"notebook is missing required HMM/dual-method content: {token}")

    if not any(token in content for token in CITATION_TOKENS):
        failures.append("notebook is missing scholarly in-text citation content")
    if not BIBLIOGRAPHY.is_file() or BIBLIOGRAPHY.stat().st_size == 0:
        failures.append(
            "canonical bibliography registry reports/references.bib is missing or empty"
        )

    for pattern in FORBIDDEN_MARKOV_RESULT_PATTERNS:
        if pattern.search(content):
            failures.append(f"notebook contains forbidden Markov-result leakage: {pattern.pattern}")

    if failures:
        raise RuntimeError("Notebook validation failed: " + "; ".join(failures))
    return _notebook_sha256(path)


def _export_notebook_html(notebook_path: Path, html_path: Path) -> None:
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = False
    exporter.exclude_output = False
    body, _ = exporter.from_filename(str(notebook_path))
    print_styles = """
<style>
@media print {
  .jp-RenderedHTMLCommon p { text-align: justify; hyphens: auto; }
  .jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2,
  .jp-RenderedHTMLCommon h3 { text-align: center; }
  .jp-RenderedHTMLCommon img { display: block; margin-left: auto; margin-right: auto; }
  .jp-RenderedHTMLCommon table { margin-left: auto; margin-right: auto; }
}
</style>
"""
    body = body.replace("</head>", print_styles + "</head>", 1)
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
        "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("Chrome did not produce a non-empty notebook PDF.")


def _template_cover(template_path: Path) -> object:
    template = pymupdf.open(template_path)
    try:
        if not template.page_count:
            raise RuntimeError("The supplied PDF template has no pages.")
        template_cover = template[0]
        for member in (REMOVED_MEMBER, *REMAINING_MEMBERS):
            matches = template_cover.search_for(member)
            if matches and len(matches) != 2:
                raise RuntimeError(
                    f"Expected {member!r} to appear exactly twice on the template cover."
                )
            for match in matches:
                template_cover.add_redact_annot(match, fill=(1, 1, 1))
        template_cover.apply_redactions()
        sanitized_template = template.tobytes(garbage=4, deflate=True)
    finally:
        template.close()

    reader = PdfReader(BytesIO(sanitized_template))
    if not reader.pages:
        raise RuntimeError("The supplied PDF template has no pages.")

    cover = reader.pages[0]
    width = float(cover.mediabox.width)
    height = float(cover.mediabox.height)
    if width <= 0 or height <= 0:
        raise RuntimeError("The supplied PDF template cover has invalid page dimensions.")

    overlay_bytes = BytesIO()
    overlay = canvas.Canvas(overlay_bytes, pagesize=(width, height))

    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont("Helvetica-Bold", 12)
    overlay.drawString(width * 0.38, height * 0.945, GROUP_WORK_PROJECT_NUMBER)
    overlay.drawString(width * 0.33, height * 0.925, GROUP_NUMBER)
    overlay.setFont("Helvetica", 10)
    overlay.drawString(76, 646.5, REMAINING_MEMBERS[0])
    overlay.drawString(76, 622.5, REMAINING_MEMBERS[1])
    overlay.drawString(179, 483, REMAINING_MEMBERS[0])
    overlay.drawString(179, 457, REMAINING_MEMBERS[1])

    overlay.save()
    overlay_bytes.seek(0)
    cover.merge_page(PdfReader(overlay_bytes).pages[0])
    return cover


def _validate_pdf_metadata(pdf_path: Path, notebook_sha: str, expected_body_pages: int) -> None:
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("Generated report is empty.")
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) != expected_body_pages + 1:
        raise RuntimeError(
            "Generated report must contain one template cover plus notebook pages only."
        )
    metadata = reader.metadata or {}
    if metadata.get("/NotebookSHA256") != notebook_sha:
        raise RuntimeError(
            "Generated report notebook SHA-256 metadata does not match source bytes."
        )
    if metadata.get("/ArtifactRole") != "notebook-sidecar":
        raise RuntimeError("Generated report is missing notebook-sidecar metadata.")
    if metadata.get("/SourceOfTruth") != "notebooks/gwp2_vix_regime_allocation.ipynb":
        raise RuntimeError("Generated report has an invalid source-of-truth metadata path.")


def build_report(
    notebook_path: Path = NOTEBOOK,
    template_path: Path = TEMPLATE,
    output_path: Path = OUTPUT,
) -> Path:
    """Render a validated executed notebook after exactly page 1 of the supplied template."""
    notebook_sha = _validate_notebook(notebook_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gwp2-report-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        html_path = temp_dir / "notebook.html"
        notebook_pdf = temp_dir / "notebook.pdf"
        _export_notebook_html(notebook_path, html_path)
        _print_html_to_pdf(html_path, notebook_pdf)
        if not notebook_pdf.is_file() or notebook_pdf.stat().st_size == 0:
            raise RuntimeError("Notebook rendering produced an empty PDF.")

        body = PdfReader(str(notebook_pdf))
        if not body.pages:
            raise RuntimeError("Notebook rendering produced no PDF pages.")

        writer = PdfWriter()
        writer.add_page(_template_cover(template_path))
        for page in body.pages:
            writer.add_page(page)

        writer.add_metadata(
            {
                "/Title": "MScFE 622 Group Work Project 2 — VIX Regime Allocation",
                "/Subject": "Derived PDF sidecar of the canonical executed notebook",
                "/ArtifactRole": "notebook-sidecar",
                "/SourceOfTruth": "notebooks/gwp2_vix_regime_allocation.ipynb",
                "/NotebookSHA256": notebook_sha,
            }
        )
        with output_path.open("wb") as handle:
            writer.write(handle)

        _validate_pdf_metadata(output_path, notebook_sha, expected_body_pages=len(body.pages))

    return output_path


def main() -> None:
    report = build_report()
    print(f"Generated {report.relative_to(ROOT)} ({report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
