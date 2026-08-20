from __future__ import annotations

import re
from pathlib import Path

PATH = Path("BACKLOG.md")
text = PATH.read_text(encoding="utf-8")


def replace_pattern(pattern: str, replacement: str, expected: int = 1) -> None:
    global text
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != expected:
        raise SystemExit(f"Expected {expected} replacements for {pattern!r}, found {count}.")
    text = updated


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def rename_pr(number: int, old_title: str, new_title: str) -> None:
    code = f"{number:02d}"
    old_branch = f"pr-{code}-{slug(old_title)}"
    new_branch = f"pr-{code}-{slug(new_title)}"
    replace_pattern(
        rf"^## PR-{code} — {re.escape(old_title)}$",
        f"## PR-{code} — {new_title}",
    )
    replace_pattern(
        rf"^\*\*Git branch:\*\* `{re.escape(old_branch)}`$",
        f"**Git branch:** `{new_branch}`",
    )
    replace_pattern(
        rf"^\*\*Git status:\*\* `git status --short --branch` must show `{re.escape(old_branch)}` and no staged, modified, or untracked files immediately before commit and merge\.$",
        f"**Git status:** `git status --short --branch` must show `{new_branch}` and no staged, modified, or untracked files immediately before commit and merge.",
    )
    replace_pattern(
        rf"^\*\*Commit message:\*\* `PR-{code} — {re.escape(old_title)}`$",
        f"**Commit message:** `PR-{code} — {new_title}`",
    )


replace_pattern(
    r"^8\. Notebook technical prose explains finance/statistics, not Python library names\. Standalone PDF prose is non-technical and avoids model/algorithm/library names\.$",
    "8. Notebook technical prose explains finance/statistics rather than narrating Python library calls. The PDF is a rendered sidecar of the canonical executed notebook and therefore preserves the same notebook explanations, equations, code cells, stored outputs, tables, figures, citations, and Works Cited; PDF-only analytical narrative is forbidden.",
)
replace_pattern(
    r"^10\. Before equations containing Greek symbols, list each Greek letter and pronunciation\. The technical notebook and standalone PDF must support external theoretical, methodological, and empirical claims with verifiable scholarly sources using MLA 9 in-text citations and a Works Cited section\. Official primary sources may supplement scholarly literature for data/index definitions but do not replace academic support\. Every citation must resolve to the canonical `reports/references\.bib`; bare URLs are not citations and fabricated bibliographic metadata is forbidden\.$",
    "10. Before equations containing Greek symbols, list each Greek letter and pronunciation. The technical notebook must support external theoretical, methodological, and empirical claims with verifiable scholarly sources using MLA 9 in-text citations and a Works Cited section; the PDF sidecar inherits that exact cited content from the notebook and may not diverge. Official primary sources may supplement scholarly literature for data/index definitions but do not replace academic support. Every citation must resolve to the canonical `reports/references.bib`; bare URLs are not citations and fabricated bibliographic metadata is forbidden.",
)
replace_pattern(
    r"^The standalone PDF remains non-technical in its narrative, but it also uses MLA 9 in-text citations for externally sourced factual or methodological claims, source notes for externally derived data/figures, and a final \*\*Works Cited\*\* section\. Bibliographic titles may contain technical terminology; the no-model/no-algorithm wording rule applies to report narrative, not to bibliographic metadata\.$",
    "The PDF report is a rendered sidecar of the canonical executed notebook. It inherits the notebook's MLA 9 in-text citations, source notes, equations, code cells, stored outputs, tables, figures, interpretations, limitations, and final **Works Cited** in notebook order. The supplied template contributes page 1 as the cover; template instruction page 2 is excluded. No separately authored PDF narrative or independent analysis is permitted.",
)
replace_pattern(
    r"^README has exact technical parity from canonical artifacts\. HTML `reports/gwp2_vix_regime_allocation\.html` is exported from stored notebook outputs without execution/refitting and preserves the notebook citations/Works Cited\. Separate PDF `reports/Stochastic_Modeling_GWP2_Report\.pdf` is non-technical/no-code, uses provided template \*\*page1 only\*\*, excludes instruction page2, preserves known team names/blank unknown fields, has decision parity, and contains MLA 9 in-text citations/source notes plus a Works Cited derived from the same canonical registry; render every final page for visual QA\.$",
    "README has exact technical parity from canonical artifacts. HTML `reports/gwp2_vix_regime_allocation.html` is exported from stored notebook outputs without execution/refitting and preserves the notebook citations/Works Cited. Separate PDF sidecar `reports/Stochastic_Modeling_GWP2_Report.pdf` uses provided template **page1 only**, excludes instruction page2, preserves known team names/blank unknown fields, and then renders the canonical executed notebook in order. The PDF stores the source notebook path and exact notebook SHA-256 for stale-sidecar detection; render every final page for visual QA.",
)
replace_pattern(
    r"^Final ZIP `dist/MScFE_622_GWP2_submission\.zip` contains exactly notebook, HTML, README, `pyproject\.toml`, `reports/references\.bib`, `data/processed/step1_data\.csv`, and `src/vix_regime_allocation/\*\*/\*\.py`\. It excludes standalone PDF, `\.git`, `\.github`, tests, rendered QA, caches, coverage, `\.env\*`, keys\. Sorted POSIX members, no symlinks/traversal, timestamp `1980-01-01`\. Submission manifest keys: `schema_version,zip_path,zip_sha256,standalone_pdf_path,standalone_pdf_sha256,included_files,member_sha256`; no timestamp\. Standalone PDF is uploaded separately\.$",
    "Final ZIP `dist/MScFE_622_GWP2_submission.zip` contains exactly notebook, HTML, README, `pyproject.toml`, `reports/references.bib`, `data/processed/step1_data.csv`, and `src/vix_regime_allocation/**/*.py`. It excludes the separately uploaded PDF sidecar, `.git`, `.github`, tests, rendered QA, caches, coverage, `.env*`, keys. Sorted POSIX members, no symlinks/traversal, timestamp `1980-01-01`. Submission manifest keys remain `schema_version,zip_path,zip_sha256,standalone_pdf_path,standalone_pdf_sha256,included_files,member_sha256`; no timestamp. The PDF sidecar is uploaded separately.",
)

rename_pr(28, "Non-technical PDF builder", "Notebook PDF sidecar builder")
rename_pr(29, "Generate/visually verify Step1–4 PDF", "Generate/visually verify Step1–4 PDF sidecar")
rename_pr(44, "Extend PDF builder through Step5", "Extend PDF sidecar builder through Step5")
rename_pr(45, "Final Step1–5 nontechnical PDF + visual QA", "Final Step1–5 PDF sidecar + visual QA")

replace_pattern(r"^reports/Stochastic_Modeling_GWP2_Report\.md$\n", "", expected=2)

TASK_REPLACEMENTS = {
    "T28.1": "- [ ] T28.1 Build from template page1 plus the canonical executed notebook; reject notebook error outputs; render notebook cells/outputs in order; exclude template page2; preserve `reports/references.bib`-resolved citations/Works Cited; embed sidecar role, source notebook path, and exact notebook SHA-256; forbid PDF-only analysis.",
    "T28.2": "- [ ] T28.2 Add offline tests for cover/page2 exclusion, notebook error rejection, notebook-SHA/source-path metadata, non-empty renderability, known names, inherited `reports/references.bib` citation integrity, and rejection of any stale sidecar hash.",
    "AC28.1": "- [ ] AC28.1 (`T28.1`) Generated fixture PDF is a notebook-derived sidecar with correct cover/no page2, exact notebook provenance metadata, no independent analysis path, and the notebook's resolved scholarly citations/source notes/Works Cited.",
    "AC28.2": "- [ ] AC28.2 (`T28.2`) All PDF sidecar, provenance, stale-hash, renderability, and citation-integrity tests pass offline.",
    "T29.1": "- [ ] T29.1 Generate the Step1–4 PDF sidecar from the committed executed notebook and template page1 only; preserve notebook equations, project-function cells, stored outputs, figures, explanations, limitations, MLA 9 citations, source notes, and Works Cited resolved through `reports/references.bib`; add no PDF-only prose.",
    "T29.2": "- [ ] T29.2 Render/inspect every page for clipping/overlap/glyph/blank/split defects, verify the stored notebook SHA-256 metadata, and mark the sidecar interim until Step5.",
    "AC29.1": "- [ ] AC29.1 (`T29.1`) PDF content is derived from the executed notebook in order, includes the notebook's technical content and resolved citations/source notes/Works Cited, has the correct cover/no page2, and contains no independently authored analytical section.",
    "AC29.2": "- [ ] AC29.2 (`T29.2`) Every page passes visual QA, notebook provenance metadata matches exactly, and Step5 sidecar regeneration requirement is explicit.",
    "T31.1": "- [ ] T31.1 Validate manifest/input hash/artifacts plus notebook/README exact technical parity, HTML notebook hash, PDF notebook-content parity and exact notebook-SHA/source-path provenance, and citation integrity against `reports/references.bib`: resolved in-text cites, cited-only Works Cited entries, required scholarly support, and source notes.",
    "AC31.1": "- [ ] AC31.1 (`T31.1`) Any missing/stale/mismatched canonical technical artifact, notebook/PDF provenance mismatch, or citation/source-note/Works-Cited defect fails at the correct parity level.",
    "T44.1": "- [ ] T44.1 Extend PDF-sidecar validation through the final Step5 notebook sections and `reports/references.bib`; require the benchmark comparison, summary, sensitivity, recommendation, limitations, figures, equations, stored outputs, resolved MLA 9 citations/source notes/Works Cited to come from the executed notebook with no second analysis.",
    "T44.2": "- [ ] T44.2 Add tests for Step5 stale/missing notebook content, exact notebook-SHA/source-path metadata, final citation integrity, template cover/page2 exclusion, known names, and rejection of PDF-only analytical content.",
    "AC44.1": "- [ ] AC44.1 (`T44.1`) Builder requires the exact final executed notebook as its Step5 analytical source and a valid canonical scientific-source registry with resolved rendered citations.",
    "T45.1": "- [ ] T45.1 Generate the final Step1–5 PDF sidecar from the fully executed notebook with canonical metrics, cumulative comparison figure, sensitivity, both-benchmark conclusion, recommendation, costs, full-sample/non-OOS limitations, equations, code/output cells, MLA 9 citations, official-primary source notes, and final Works Cited from `reports/references.bib`; use template page1 only and no PDF-only narrative.",
    "AC45.1": "- [ ] AC45.1 (`T45.1`) Final PDF is an exact rendered-notebook sidecar after the template cover, has matching notebook SHA-256/source-path metadata, includes the notebook's technical content and limitations, and preserves resolved scholarly citations/source notes with cited-only Works Cited entries.",
    "T47.1": "- [ ] T47.1 Extend checker/tests to Step5 manifest/artifacts, notebook/README technical parity, current HTML, PDF exact notebook-sidecar parity with notebook-SHA/source-path provenance, and final citation integrity against `reports/references.bib` for notebook-derived HTML/PDF artifacts.",
    "AC47.1": "- [ ] AC47.1 (`T47.1`) Every Step5 stale/missing/hash/value sidecar defect, notebook/PDF provenance mismatch, and every unresolved/duplicate/orphan/URL-only citation or missing Works Cited/source-note defect fails at the correct parity level.",
}

for identifier, replacement in TASK_REPLACEMENTS.items():
    replace_pattern(rf"^- \[ \] {re.escape(identifier)} .*$", replacement)

replace_pattern(
    r"^- \[ \] Final notebook fully executed with resolved MLA 9 scholarly citations/source notes/final Works Cited; README exact technical parity; HTML exact duplicate; separate PDF nontechnical decision parity with resolved scholarly citations/source notes/Works Cited, template page1 only, and visual QA\.$",
    "- [ ] Final notebook fully executed with resolved MLA 9 scholarly citations/source notes/final Works Cited; README exact technical parity; HTML exact duplicate; separate PDF exact rendered-notebook sidecar with matching notebook SHA-256/source-path provenance, template page1 only, and visual QA.",
)

PATH.write_text(text, encoding="utf-8")
