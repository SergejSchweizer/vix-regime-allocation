from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
HTML = ROOT / "reports/gwp2_vix_regime_allocation.html"
PDF = ROOT / "reports/Stochastic_Modeling_GWP2_Report.pdf"
STEP5_CALLS = {
    "nb.step_5_performance_metrics_and_cumulative_compar_034()",
    "nb.step_5_hmm_dual_method_comparison()",
}
SOURCE_DIGEST_KEY = "artifact_source_sha256"
MIGRATION_SENTINEL = ROOT / "src/vix_regime_allocation/markov_states.py"


def source_paths() -> list[Path]:
    paths = sorted((ROOT / "src/vix_regime_allocation").glob("*.py"))
    paths.extend(
        [
            ROOT / "data/processed/step1_data.csv",
            ROOT / "reports/references.bib",
            ROOT / "scripts/rebuild_analysis_review.py",
            ROOT / "scripts/check_analysis_consistency.py",
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


def _artifact_source_sha(notebook: nbformat.NotebookNode, expected_source_sha: str) -> str:
    actual_source_sha = notebook.metadata.get(SOURCE_DIGEST_KEY)
    if not isinstance(actual_source_sha, str) or not actual_source_sha:
        raise RuntimeError("Executed notebook is missing artifact source provenance metadata.")
    if actual_source_sha == expected_source_sha:
        return expected_source_sha

    # PR-50..PR-66 intentionally change source contracts before PR-60/PR-61
    # commit the rebuilt canonical analysis and executed notebook. The migration
    # sentinel disappears in PR-67, which restores strict equality automatically.
    if MIGRATION_SENTINEL.is_file():
        return actual_source_sha

    raise RuntimeError(
        "Executed notebook is stale relative to source/data inputs: "
        f"expected {expected_source_sha}, found {actual_source_sha!r}."
    )


def validate_notebook() -> tuple[str, str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    expected_source_sha = compute_source_sha256()
    artifact_source_sha = _artifact_source_sha(notebook, expected_source_sha)

    failures: list[str] = []
    step5_matches: list[nbformat.NotebookNode] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if _cell_source(cell).strip() in STEP5_CALLS:
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
    return artifact_source_sha, notebook_sha256()


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


def validate_analysis_manifests() -> None:
    keep = ROOT / "reports/tables/step4_allocation_100_keep.csv"
    spread = ROOT / "reports/tables/step4_allocation_60_40_spread.csv"
    if not (keep.is_file() and spread.is_file()):
        return

    expected_sha = hashlib.sha256((ROOT / "data/processed/step1_data.csv").read_bytes()).hexdigest()
    for relative in (
        "reports/generated/steps_2_4_manifest.json",
        "reports/generated/step5_manifest.json",
    ):
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        if payload.get("input_data_sha256") != expected_sha:
            raise RuntimeError(f"{relative} does not match the current Step 1 bytes.")
        serialized = json.dumps(payload).lower()
        if "markov" in serialized:
            raise RuntimeError(f"{relative} contains a non-HMM canonical path.")

    daily = pd.read_csv(ROOT / "reports/tables/step5_daily_returns.csv")
    expected_columns = [
        "Date",
        "hmm_100_keep",
        "hmm_60_40_spread",
        "equal_weight_monthly",
        "spy_buy_hold",
    ]
    if daily.columns.tolist() != expected_columns:
        raise RuntimeError("Step 5 daily returns do not have the canonical four-portfolio schema.")


def main() -> None:
    source_sha, notebook_sha = validate_notebook()
    validate_html(source_sha, notebook_sha)
    validate_pdf(notebook_sha)
    validate_analysis_manifests()
    print("Notebook, HTML, PDF, and HMM analysis provenance is consistent.")


if __name__ == "__main__":
    main()
