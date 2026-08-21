from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat
import pytest
from pypdf import PdfReader, PdfWriter

from scripts import build_pdf_report as report


def _valid_notebook() -> nbformat.NotebookNode:
    narrative = """# MScFE 622 — VIX Regime Allocation

## Hidden Markov Model and Expectation-Maximization
The Gaussian Hidden Markov Model is estimated with the Baum-Welch form of the
Expectation-Maximization procedure (Baum et al. 164–171; Rabiner 257–286).
Viterbi decoding follows the dynamic-programming principle described by Viterbi
(260–269). Model selection uses AIC and BIC (Akaike 716–723; Schwarz 461–464).

## Step 4 — 100% Keep
The canonical 100% Keep allocation is `100_keep`.

## Step 4 — 60/40 Spread
The canonical 60/40 Spread allocation is `60_40_spread`.

## Step 5 — four portfolios
The exact portfolios are `hmm_100_keep`, `hmm_60_40_spread`,
`equal_weight_monthly`, and `spy_buy_hold`.

## Works Cited
Baum, Leonard E., et al. “A Maximization Technique...”
Rabiner, Lawrence R. “A Tutorial on Hidden Markov Models...”
Viterbi, Andrew J. “Error Bounds for Convolutional Codes...”
Akaike, Hirotugu. “A New Look at the Statistical Model Identification.”
Schwarz, Gideon. “Estimating the Dimension of a Model.”
"""
    cells = [nbformat.v4.new_markdown_cell(narrative)]
    cells.extend(
        nbformat.v4.new_code_cell(source, execution_count=index, outputs=[])
        for index, source in enumerate(
            (
                "from vix_regime_allocation import notebook_helpers as nb",
                "from vix_regime_allocation import notebook_sensitivity as sensitivity_nb",
                *report.REQUIRED_CODE_CALLS,
            ),
            start=1,
        )
    )
    return nbformat.v4.new_notebook(cells=cells)


def _write_notebook(path: Path, notebook: nbformat.NotebookNode | None = None) -> Path:
    nbformat.write(notebook or _valid_notebook(), path)
    return path


def _write_pdf(path: Path, pages: int = 1, metadata: dict[str, str] | None = None) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    if metadata:
        writer.add_metadata(metadata)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def _stub_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report,
        "_export_notebook_html",
        lambda _notebook, html: html.write_text("<html></html>", encoding="utf-8"),
    )
    monkeypatch.setattr(report, "_print_html_to_pdf", lambda _html, pdf: _write_pdf(pdf))


def test_build_report_uses_template_page_one_and_exact_notebook_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _write_notebook(tmp_path / "report.ipynb")
    template = _write_pdf(tmp_path / "template.pdf", pages=2)
    output = tmp_path / "report.pdf"
    _stub_render(monkeypatch)

    result = report.build_report(notebook, template, output)

    assert result == output
    reader = PdfReader(str(output))
    assert len(reader.pages) == 2, "template instruction pages must not be appended"
    expected_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert reader.metadata is not None
    assert reader.metadata.get("/NotebookSHA256") == expected_sha
    assert reader.metadata.get("/ArtifactRole") == "notebook-sidecar"


def test_unexecuted_code_cell_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    notebook.cells[1].execution_count = None
    path = _write_notebook(tmp_path / "unexecuted.ipynb", notebook)
    with pytest.raises(RuntimeError, match="not executed"):
        report._validate_notebook(path)


def test_failed_code_cell_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    notebook.cells[1].outputs = [
        nbformat.v4.new_output(
            "error", ename="RuntimeError", evalue="boom", traceback=["RuntimeError: boom"]
        )
    ]
    path = _write_notebook(tmp_path / "failed.ipynb", notebook)
    with pytest.raises(RuntimeError, match="RuntimeError: boom"):
        report._validate_notebook(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("Hidden Markov Model", "latent-state model", "Hidden Markov Model"),
        ("Expectation-Maximization", "iterative fitting", "Expectation-Maximization"),
        ("Baum-Welch", "iterative recursion", "Baum-Welch"),
        ("100% Keep", "top asset", "100% Keep"),
        ("60/40 Spread", "two assets", "60/40 Spread"),
        ("Works Cited", "References", "Works Cited"),
    ],
)
def test_missing_required_notebook_sections_are_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    notebook = _valid_notebook()
    notebook.cells[0].source = notebook.cells[0].source.replace(old, new)
    path = _write_notebook(tmp_path / "missing.ipynb", notebook)
    with pytest.raises(RuntimeError, match=message.replace("%", "%")):
        report._validate_notebook(path)


def test_markov_result_leakage_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    notebook.cells.insert(1, nbformat.v4.new_markdown_cell("Selected model: Markov K=2"))
    path = _write_notebook(tmp_path / "markov.ipynb", notebook)
    with pytest.raises(RuntimeError, match="forbidden Markov-result leakage"):
        report._validate_notebook(path)


def test_missing_four_portfolio_contract_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    notebook.cells[0].source = notebook.cells[0].source.replace("hmm_60_40_spread", "other")
    path = _write_notebook(tmp_path / "portfolios.ipynb", notebook)
    with pytest.raises(RuntimeError, match="hmm_60_40_spread"):
        report._validate_notebook(path)


def test_rendered_outputs_participate_in_notebook_content_validation(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    portfolio_tokens = (
        "hmm_100_keep",
        "hmm_60_40_spread",
        "equal_weight_monthly",
        "spy_buy_hold",
    )
    for token in portfolio_tokens:
        notebook.cells[0].source = notebook.cells[0].source.replace(token, "portfolio")
    target = next(
        cell
        for cell in notebook.cells
        if getattr(cell, "source", "") == "nb.step_5_hmm_dual_method_comparison()"
    )
    target.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"text/plain": "\n".join(portfolio_tokens)},
            metadata={},
        )
    ]
    path = _write_notebook(tmp_path / "outputs.ipynb", notebook)
    assert report._validate_notebook(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_scholarly_citations_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    for token in report.CITATION_TOKENS:
        notebook.cells[0].source = notebook.cells[0].source.replace(token, "Source")
    path = _write_notebook(tmp_path / "citations.ipynb", notebook)
    with pytest.raises(RuntimeError, match="scholarly in-text citation"):
        report._validate_notebook(path)


def test_missing_bibliography_registry_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _write_notebook(tmp_path / "bibliography.ipynb")
    monkeypatch.setattr(report, "BIBLIOGRAPHY", tmp_path / "missing-references.bib")
    with pytest.raises(RuntimeError, match="bibliography registry"):
        report._validate_notebook(notebook)


def test_missing_required_helper_call_is_rejected(tmp_path: Path) -> None:
    notebook = _valid_notebook()
    notebook.cells = [
        cell
        for cell in notebook.cells
        if getattr(cell, "source", "") != "nb.step_4_dual_allocations()"
    ]
    path = _write_notebook(tmp_path / "helper.ipynb", notebook)
    with pytest.raises(RuntimeError, match="nb.step_4_dual_allocations"):
        report._validate_notebook(path)


def test_empty_rendered_body_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = _write_notebook(tmp_path / "report.ipynb")
    template = _write_pdf(tmp_path / "template.pdf")
    output = tmp_path / "report.pdf"
    monkeypatch.setattr(
        report,
        "_export_notebook_html",
        lambda _notebook, html: html.write_text("<html></html>", encoding="utf-8"),
    )
    monkeypatch.setattr(report, "_print_html_to_pdf", lambda _html, pdf: pdf.write_bytes(b""))

    with pytest.raises(RuntimeError, match="empty PDF"):
        report.build_report(notebook, template, output)


def test_empty_template_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = _write_notebook(tmp_path / "report.ipynb")
    template = _write_pdf(tmp_path / "template.pdf", pages=0)
    _stub_render(monkeypatch)
    with pytest.raises(RuntimeError, match="template has no pages"):
        report.build_report(notebook, template, tmp_path / "report.pdf")


def test_pdf_metadata_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    pdf = _write_pdf(
        tmp_path / "report.pdf",
        pages=2,
        metadata={
            "/NotebookSHA256": "wrong",
            "/ArtifactRole": "notebook-sidecar",
            "/SourceOfTruth": "notebooks/gwp2_vix_regime_allocation.ipynb",
        },
    )
    with pytest.raises(RuntimeError, match="SHA-256 metadata"):
        report._validate_pdf_metadata(pdf, "expected", expected_body_pages=1)


def test_pdf_metadata_role_or_source_mismatch_is_rejected(tmp_path: Path) -> None:
    wrong_role = _write_pdf(
        tmp_path / "wrong-role.pdf",
        pages=2,
        metadata={
            "/NotebookSHA256": "expected",
            "/ArtifactRole": "independent-report",
            "/SourceOfTruth": "notebooks/gwp2_vix_regime_allocation.ipynb",
        },
    )
    with pytest.raises(RuntimeError, match="notebook-sidecar metadata"):
        report._validate_pdf_metadata(wrong_role, "expected", expected_body_pages=1)

    wrong_source = _write_pdf(
        tmp_path / "wrong-source.pdf",
        pages=2,
        metadata={
            "/NotebookSHA256": "expected",
            "/ArtifactRole": "notebook-sidecar",
            "/SourceOfTruth": "other.ipynb",
        },
    )
    with pytest.raises(RuntimeError, match="source-of-truth"):
        report._validate_pdf_metadata(wrong_source, "expected", expected_body_pages=1)


def test_pdf_page_count_mismatch_is_rejected(tmp_path: Path) -> None:
    pdf = _write_pdf(
        tmp_path / "report.pdf",
        pages=3,
        metadata={
            "/NotebookSHA256": "expected",
            "/ArtifactRole": "notebook-sidecar",
            "/SourceOfTruth": "notebooks/gwp2_vix_regime_allocation.ipynb",
        },
    )
    with pytest.raises(RuntimeError, match="one template cover"):
        report._validate_pdf_metadata(pdf, "expected", expected_body_pages=1)
