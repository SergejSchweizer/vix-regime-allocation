from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
HTML = ROOT / "reports/gwp2_vix_regime_allocation.html"
PDF = ROOT / "reports/Stochastic_Modeling_GWP2_Report.pdf"
STEP5_CALL = "nb.step_5_performance_metrics_and_cumulative_compar_034()"
SOURCE_DIGEST_KEY = "artifact_source_sha256"


def source_paths() -> list[Path]:
    paths = sorted((ROOT / "src/vix_regime_allocation").glob("*.py"))
    paths.extend(
        [
            ROOT / "data/processed/step1_data.csv",
            ROOT / "reports/references.bib",
            ROOT / "scripts/rebuild_analysis_review.py",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing artifact source inputs: {missing}")
    return paths


def compute_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in source_paths():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def notebook_sha256() -> str:
    return hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()


def _cell_source(cell: nbformat.NotebookNode) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_notebook() -> tuple[str, str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    expected_source_sha = compute_source_sha256()
    actual_source_sha = notebook.metadata.get(SOURCE_DIGEST_KEY)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError(
            "Executed notebook is stale relative to source/data inputs: "
            f"expected {expected_source_sha}, found {actual_source_sha!r}."
        )

    failures: list[str] = []
    step5_matches: list[nbformat.NotebookNode] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if _cell_source(cell).strip() == STEP5_CALL:
            step5_matches.append(cell)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append(
                    f"cell {index}: {output.get('ename', 'Error')}: {output.get('evalue', '')}"
                )
    if failures:
        raise RuntimeError("Notebook contains failed outputs: " + "; ".join(failures))
    if len(step5_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Step 5 cumulative-output cell, found {len(step5_matches)}."
        )
    if not any(
        output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
        for output in step5_matches[0].get("outputs", [])
    ):
        raise RuntimeError("Step 5 cumulative comparison is not embedded as image/png.")

    serialized = NOTEBOOK.read_text(encoding="utf-8")
    if "/home/runner/work/" in serialized:
        raise RuntimeError("Notebook contains a runner-local absolute path.")
    if "![Cumulative performance comparison](/" in serialized:
        raise RuntimeError("Notebook contains an absolute Markdown image path.")
    return expected_source_sha, notebook_sha256()


def validate_html(source_sha: str, notebook_sha: str) -> None:
    if not HTML.is_file() or HTML.stat().st_size == 0:
        raise RuntimeError("Canonical HTML report is missing or empty.")
    prefix = HTML.read_text(encoding="utf-8", errors="strict")[:2048]
    if f"ArtifactSourceSHA256:{source_sha}" not in prefix:
        raise RuntimeError("HTML source digest does not match the notebook source inputs.")
    if f"NotebookSHA256:{notebook_sha}" not in prefix:
        raise RuntimeError("HTML notebook digest does not match the canonical notebook.")


def validate_pdf(notebook_sha: str) -> None:
    if not PDF.is_file() or PDF.stat().st_size == 0:
        raise RuntimeError("Canonical PDF report is missing or empty.")
    reader = PdfReader(str(PDF))
    metadata = reader.metadata
    if metadata is None:
        raise RuntimeError("PDF metadata is missing.")
    if metadata.get("/ArtifactRole") != "notebook-sidecar":
        raise RuntimeError("PDF is missing the notebook-sidecar artifact role.")
    if metadata.get("/SourceOfTruth") != "notebooks/gwp2_vix_regime_allocation.ipynb":
        raise RuntimeError("PDF source-of-truth metadata is invalid.")
    if metadata.get("/NotebookSHA256") != notebook_sha:
        raise RuntimeError("PDF was not built from the canonical executed notebook bytes.")


def main() -> None:
    source_sha, notebook_sha = validate_notebook()
    validate_html(source_sha, notebook_sha)
    validate_pdf(notebook_sha)
    print("Notebook, HTML, and PDF artifact provenance is consistent.")


if __name__ == "__main__":
    main()
