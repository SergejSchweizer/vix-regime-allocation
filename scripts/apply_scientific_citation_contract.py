from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "BACKLOG.md"
README = ROOT / "README.md"
BACKLOG_CHECKER = ROOT / "scripts" / "check_backlog_contract.py"
README_CHECKER = ROOT / "scripts" / "check_readme_sidecar.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}.")
    return text.replace(old, new, 1)


def section_bounds(text: str, pr_number: int) -> tuple[int, int]:
    header = re.search(rf"^## PR-{pr_number:02d} — .+$", text, flags=re.MULTILINE)
    if header is None:
        raise SystemExit(f"Missing PR-{pr_number:02d} section.")
    next_header = re.search(r"^## PR-\d{2} — .+$", text[header.end() :], flags=re.MULTILINE)
    end = header.end() + next_header.start() if next_header else len(text)
    return header.start(), end


def update_pr_section(text: str, pr_number: int, replacements: list[tuple[str, str, str]]) -> str:
    start, end = section_bounds(text, pr_number)
    section = text[start:end]
    for old, new, label in replacements:
        section = replace_once(section, old, new, f"PR-{pr_number:02d} {label}")
    return text[:start] + section + text[end:]


def add_reference_file(text: str, pr_number: int) -> str:
    start, end = section_bounds(text, pr_number)
    section = text[start:end]
    anchor = "notebooks/gwp2_vix_regime_allocation.ipynb\n"
    if "reports/references.bib" in section:
        raise SystemExit(f"PR-{pr_number:02d} already owns reports/references.bib.")
    section = replace_once(
        section,
        anchor,
        anchor + "reports/references.bib\n",
        f"PR-{pr_number:02d} reference registry ownership",
    )
    return text[:start] + section + text[end:]


def update_backlog() -> None:
    text = BACKLOG.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "10. Before equations containing Greek symbols, list each Greek letter and pronunciation. External factual claims use consulted authoritative sources with MLA in-text citations/bibliography.\n",
        "10. Before equations containing Greek symbols, list each Greek letter and pronunciation. The technical notebook and standalone PDF must support external theoretical, methodological, and empirical claims with verifiable scholarly sources using MLA 9 in-text citations and a Works Cited section. Official primary sources may supplement scholarly literature for data/index definitions but do not replace academic support. Every citation must resolve to the canonical `reports/references.bib`; bare URLs are not citations and fabricated bibliographic metadata is forbidden.\n",
        "global citation rule",
    )

    citation_contract = """## Scientific citation contract

`reports/references.bib` is the single canonical source registry. PR-05 creates it; serialized notebook PRs may extend it only when a new cited source is required. Entries use stable citation keys and complete verifiable metadata: author(s), title, venue or publisher, year, and DOI when one exists; otherwise ISBN for scholarly books or a stable official URL for primary data/index documentation. No citation metadata may be invented.

Scientific support is mandatory, not optional. Peer-reviewed journal/conference papers and scholarly books/textbooks support Markov-chain theory, HMM/EM/Viterbi/posterior inference, information criteria, performance metrics, backtesting limitations, and other methodological claims. Official primary sources such as index methodology or data-provider documentation may support definitions and provenance only. If a claim cannot be supported, omit it or label it explicitly as a project assumption/decision rule.

The notebook uses MLA 9 parenthetical citations adjacent to externally sourced definitions, equations, methodological claims, and interpretations, plus a final **Works Cited** section rendered from the canonical registry. Each major method section has at least one relevant scholarly citation. Tables and figures include concise source notes distinguishing project calculations from external data/method sources.

The standalone PDF remains non-technical in its narrative, but it also uses MLA 9 in-text citations for externally sourced factual or methodological claims, source notes for externally derived data/figures, and a final **Works Cited** section. Bibliographic titles may contain technical terminology; the no-model/no-algorithm wording rule applies to report narrative, not to bibliographic metadata.

Citation integrity is deterministic: every in-text citation in notebook/PDF must resolve to `reports/references.bib`; every entry printed in an artifact's Works Cited must be cited in that artifact; duplicate keys, unresolved cites, bibliography-only orphan entries in rendered Works Cited, and URL-only pseudo-citations fail validation. PR-31 establishes Step1–4 citation parity checks and PR-47 extends them through Step5.

"""
    text = replace_once(
        text,
        "## Reports/submission\n",
        citation_contract + "## Reports/submission\n",
        "scientific citation contract section",
    )

    text = replace_once(
        text,
        "Canonical technical notebook: `notebooks/gwp2_vix_regime_allocation.ipynb`. Each step visibly has question/step number, project-function calls, stored code output, equations/definitions, tables/plots, interpretation/recommendation, limitations, citations where applicable; execute top-to-bottom before commit.\n\nREADME has exact technical parity from canonical artifacts. HTML `reports/gwp2_vix_regime_allocation.html` is exported from stored notebook outputs without execution/refitting. Separate PDF `reports/Stochastic_Modeling_GWP2_Report.pdf` is non-technical/no-code, uses provided template **page1 only**, excludes instruction page2, preserves known team names/blank unknown fields, and has decision parity; render every final page for visual QA.\n\nFinal ZIP `dist/MScFE_622_GWP2_submission.zip` contains exactly notebook, HTML, README, `pyproject.toml`, `data/processed/step1_data.csv`, and `src/vix_regime_allocation/**/*.py`. It excludes standalone PDF, `.git`, `.github`, tests, rendered QA, caches, coverage, `.env*`, keys. Sorted POSIX members, no symlinks/traversal, timestamp `1980-01-01`. Submission manifest keys: `schema_version,zip_path,zip_sha256,standalone_pdf_path,standalone_pdf_sha256,included_files,member_sha256`; no timestamp. Standalone PDF is uploaded separately.\n",
        "Canonical technical notebook: `notebooks/gwp2_vix_regime_allocation.ipynb`. Each step visibly has question/step number, project-function calls, stored code output, equations/definitions, tables/plots, interpretation/recommendation, limitations, mandatory MLA 9 scholarly citations/source notes, and a final Works Cited rendered from `reports/references.bib`; execute top-to-bottom before commit.\n\nREADME has exact technical parity from canonical artifacts. HTML `reports/gwp2_vix_regime_allocation.html` is exported from stored notebook outputs without execution/refitting and preserves the notebook citations/Works Cited. Separate PDF `reports/Stochastic_Modeling_GWP2_Report.pdf` is non-technical/no-code, uses provided template **page1 only**, excludes instruction page2, preserves known team names/blank unknown fields, has decision parity, and contains MLA 9 in-text citations/source notes plus a Works Cited derived from the same canonical registry; render every final page for visual QA.\n\nFinal ZIP `dist/MScFE_622_GWP2_submission.zip` contains exactly notebook, HTML, README, `pyproject.toml`, `reports/references.bib`, `data/processed/step1_data.csv`, and `src/vix_regime_allocation/**/*.py`. It excludes standalone PDF, `.git`, `.github`, tests, rendered QA, caches, coverage, `.env*`, keys. Sorted POSIX members, no symlinks/traversal, timestamp `1980-01-01`. Submission manifest keys: `schema_version,zip_path,zip_sha256,standalone_pdf_path,standalone_pdf_sha256,included_files,member_sha256`; no timestamp. Standalone PDF is uploaded separately.\n",
        "reports/submission citation wording",
    )

    for pr in (5, 21, 22, 23, 24, 25, 40, 41, 42):
        text = add_reference_file(text, pr)

    text = update_pr_section(
        text,
        5,
        [
            (
                "- [ ] T05.1 Add complete Step1 technical section using project functions: data checks, equations/notation, both plots, assumptions/limitations, interpretation, MLA citations when used.\n- [ ] T05.2 Execute notebook with stored outputs and synchronize README Step1 status/paths without recomputation.\n\n### Acceptance criteria\n\n- [ ] AC05.1 (`T05.1`) Notebook covers all Step1 deliverables, uses shared functions, correct notation, evidence-bounded interpretation, and no failed/unexecuted cells.\n- [ ] AC05.2 (`T05.2`) README accurately references canonical Step1 artifacts and claims no later-step result.\n",
                "- [ ] T05.1 Add complete Step1 technical section using project functions: data checks, equations/notation, both plots, assumptions/limitations, interpretation, MLA 9 in-text citations for external claims, and source notes for external data/figures.\n- [ ] T05.2 Create `reports/references.bib` as the canonical verified source registry with unique keys, complete scholarly metadata, and official primary data/index sources kept distinct from scholarly support; render the notebook Works Cited from cited entries only.\n- [ ] T05.3 Execute notebook with stored outputs and synchronize README Step1 status/paths/scientific-citation policy without recomputation.\n\n### Acceptance criteria\n\n- [ ] AC05.1 (`T05.1`) Notebook covers all Step1 deliverables, uses shared functions/correct notation, has evidence-bounded interpretation, source notes, and resolved MLA 9 citations adjacent to external claims.\n- [ ] AC05.2 (`T05.2`) Canonical registry contains no duplicate keys or fabricated metadata; every rendered Works Cited entry is cited and every notebook citation resolves to the registry.\n- [ ] AC05.3 (`T05.3`) Notebook has no failed/unexecuted cells; README accurately references canonical Step1 artifacts/source policy and claims no later-step result.\n",
                "tasks and acceptance criteria",
            )
        ],
    )

    text = update_pr_section(
        text,
        21,
        [
            (
                "- [ ] T21.1 Add complete K=2/K=3 Markov technical analysis with equations/assumptions/limitations and canonical state figure using shared functions.",
                "- [ ] T21.1 Add complete K=2/K=3 Markov technical analysis with equations/assumptions/limitations, canonical state figure using shared functions, scholarly Markov-chain citations from `reports/references.bib`, and figure/data source notes.",
                "T21.1",
            ),
            (
                "- [ ] AC21.1 (`T21.1`) Notebook outputs/equations/figure are complete, correct, and evidence-bounded.",
                "- [ ] AC21.1 (`T21.1`) Notebook outputs/equations/figure are complete, correct, evidence-bounded, and all external Markov claims/caption source notes have resolved scholarly/primary citations.",
                "AC21.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        22,
        [
            (
                "- [ ] T22.1 Add complete K=2/K=3 HMM parameters/diagnostics/EM-Viterbi-posterior explanation and both canonical figures.",
                "- [ ] T22.1 Add complete K=2/K=3 HMM parameters/diagnostics/EM-Viterbi-posterior explanation and both canonical figures, citing scholarly HMM/EM/decoding/posterior sources from `reports/references.bib` with figure/data source notes.",
                "T22.1",
            ),
            (
                "- [ ] AC22.1 (`T22.1`) Notebook technical outputs/explanation/figures satisfy fixed contracts.",
                "- [ ] AC22.1 (`T22.1`) Notebook technical outputs/explanation/figures satisfy fixed contracts and every externally sourced HMM/EM/decoding/posterior claim resolves to scholarly bibliography metadata.",
                "AC22.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        23,
        [
            (
                "- [ ] T23.1 Add IC equations/four-row table, within-family comparison caveat/winners, preferred-method result/reason.",
                "- [ ] T23.1 Add IC equations/four-row table, within-family comparison caveat/winners, preferred-method result/reason, and MLA 9 citations to scholarly information-criterion sources from `reports/references.bib`.",
                "T23.1",
            ),
            (
                "- [ ] AC23.1 (`T23.1`) Displayed/CSV comparison and selected decision match shared functions with no cross-family IC claim.",
                "- [ ] AC23.1 (`T23.1`) Displayed/CSV comparison and selected decision match shared functions with no cross-family IC claim; AIC/BIC definitions and external selection claims have resolved scholarly citations.",
                "AC23.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        24,
        [
            (
                "- [ ] T24.2 Interpret states from displayed evidence; state units, ddof=1, non-annualization/sample-size limitations; execute full notebook.",
                "- [ ] T24.2 Interpret states from displayed evidence; state units, ddof=1, non-annualization/sample-size limitations; cite every external statistical/financial interpretation from `reports/references.bib`; execute full notebook.",
                "T24.2",
            ),
            (
                "- [ ] AC24.2 (`T24.2`) Interpretation is evidence-based with all fixed statistical caveats; notebook fully executed.",
                "- [ ] AC24.2 (`T24.2`) Interpretation is evidence-based with all fixed statistical caveats; external interpretation claims have resolved scholarly citations and notebook is fully executed.",
                "AC24.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        25,
        [
            (
                "- [ ] T25.2 Add evidence-supported practical takeaways/citations; write exact deterministic Steps2–4 manifest; execute/verify full Step1–4 notebook/artifacts.",
                "- [ ] T25.2 Add evidence-supported practical takeaways with MLA 9 citations, refresh `reports/references.bib`, render a complete Step1–4 Works Cited from cited entries only, verify citation/source-note integrity, write the exact deterministic Steps2–4 manifest, and execute/verify full Step1–4 notebook/artifacts.",
                "T25.2",
            ),
            (
                "- [ ] AC25.2 (`T25.2`) Citations are non-fabricated; manifest/hash/path coverage is exact; notebook/artifacts are fully consistent.",
                "- [ ] AC25.2 (`T25.2`) All notebook citations resolve to verified bibliography entries, every rendered Works Cited entry is cited, source notes are present, manifest/hash/path coverage is exact, and notebook/artifacts are fully consistent.",
                "AC25.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        28,
        [
            (
                "- [ ] T28.1 Build from template page1 + validated no-code non-technical body/canonical artifacts; exclude template page2; verify known names and render pages.",
                "- [ ] T28.1 Build from template page1 + validated no-code non-technical body/canonical artifacts and `reports/references.bib`; render MLA 9 in-text citations, source notes, and Works Cited; exclude template page2; verify known names and render pages.",
                "T28.1",
            ),
            (
                "- [ ] T28.2 Add offline tests for cover/page2/code prohibition/required sections/artifact parity/names/non-empty renderability.",
                "- [ ] T28.2 Add offline tests for cover/page2/code prohibition/required sections/artifact parity/names/non-empty renderability plus duplicate/unresolved/orphan citation failures and the bibliography-title exception to narrative technical-term checks.",
                "T28.2",
            ),
            (
                "- [ ] AC28.1 (`T28.1`) Generated fixture PDF obeys cover/nontechnical/canonical-artifact rules with no estimation path.",
                "- [ ] AC28.1 (`T28.1`) Generated fixture PDF obeys cover/nontechnical/canonical-artifact rules with no estimation path and contains resolved scholarly citations/source notes/Works Cited from the canonical registry.",
                "AC28.1",
            ),
            (
                "- [ ] AC28.2 (`T28.2`) All PDF builder tests pass offline.",
                "- [ ] AC28.2 (`T28.2`) All PDF builder and citation-integrity tests pass offline.",
                "AC28.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        29,
        [
            (
                "- [ ] T29.1 Write original non-technical Step1–4 results/recommendation/factors/limitations with canonical values/figures and consulted MLA sources; generate correct cover/no page2.",
                "- [ ] T29.1 Write original non-technical Step1–4 results/recommendation/factors/limitations with canonical values/figures, MLA 9 in-text citations to scholarly sources, official-primary data source notes, and a Works Cited derived from `reports/references.bib`; generate correct cover/no page2.",
                "T29.1",
            ),
            (
                "- [ ] AC29.1 (`T29.1`) Prose has decision parity, no code/model-algorithm-library wording, correct citations/cover.",
                "- [ ] AC29.1 (`T29.1`) Prose has decision parity, no code/model-algorithm-library wording outside bibliographic metadata, correct cover, resolved citations/source notes, and no uncited rendered Works Cited entry.",
                "AC29.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        30,
        [
            (
                "- [ ] T30.1 Export stored-output notebook to canonical HTML without execution/refit; reject failed/unexecuted/missing expected output; embed notebook SHA marker.",
                "- [ ] T30.1 Export stored-output notebook to canonical HTML without execution/refit; reject failed/unexecuted/missing expected output or missing final Works Cited; preserve MLA citations/source notes rendered from `reports/references.bib`; embed notebook SHA marker.",
                "T30.1",
            ),
            (
                "- [ ] AC30.1 (`T30.1`) HTML contains Step1–4 stored outputs/current hash and exporter has no execution path.",
                "- [ ] AC30.1 (`T30.1`) HTML contains Step1–4 stored outputs/current hash plus notebook citations/source notes/Works Cited and exporter has no execution path.",
                "AC30.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        31,
        [
            (
                "- [ ] T31.1 Validate manifest/input hash/artifacts plus notebook/README exact technical parity, HTML notebook hash, and PDF decision parity including state provenance.",
                "- [ ] T31.1 Validate manifest/input hash/artifacts plus notebook/README exact technical parity, HTML notebook hash, PDF decision parity/state provenance, and citation integrity against `reports/references.bib`: resolved in-text cites, cited-only Works Cited entries, required scholarly support, and source notes.",
                "T31.1",
            ),
            (
                "- [ ] T31.2 Add deterministic stale/missing/hash/value failure tests for every sidecar/artifact class.",
                "- [ ] T31.2 Add deterministic stale/missing/hash/value failure tests for every sidecar/artifact class plus duplicate bibliography keys, unresolved citations, orphan Works Cited entries, missing scholarly support, and URL-only pseudo-citations.",
                "T31.2",
            ),
            (
                "- [ ] AC31.1 (`T31.1`) Any missing/stale/mismatched canonical technical/decision artifact fails at correct parity level.",
                "- [ ] AC31.1 (`T31.1`) Any missing/stale/mismatched canonical technical/decision artifact or citation/source-note/Works-Cited defect fails at the correct parity level.",
                "AC31.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        32,
        [
            (
                "- [ ] T32.2 Update README/checker to document/require new jobs and Step1–4 parity policy.",
                "- [ ] T32.2 Update README/checker to document/require new jobs, Step1–4 parity policy, canonical `reports/references.bib`, MLA 9 scholarly-citation requirements, and citation-integrity enforcement.",
                "T32.2",
            ),
            (
                "- [ ] AC32.2 (`T32.2`) README/checker accurately enforce current parity/CI paths.",
                "- [ ] AC32.2 (`T32.2`) README/checker accurately enforce current parity/CI paths and scientific-citation policy.",
                "AC32.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        40,
        [
            (
                "- [ ] T40.2 Explain monthly benchmark convention, gross costs, and all full-sample/Viterbi/allocation lookahead caveats; execute full notebook.",
                "- [ ] T40.2 Explain monthly benchmark convention, gross costs, and all full-sample/Viterbi/allocation lookahead caveats with scholarly backtesting/portfolio-method citations from `reports/references.bib`; maintain source notes and execute full notebook.",
                "T40.2",
            ),
            (
                "- [ ] AC40.2 (`T40.2`) All assumptions/non-OOS caveats are explicit and full notebook has no failed/unexecuted cell.",
                "- [ ] AC40.2 (`T40.2`) All assumptions/non-OOS caveats are explicit, externally sourced methodological claims have resolved scholarly citations/source notes, and the full notebook has no failed/unexecuted cell.",
                "AC40.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        41,
        [
            (
                "- [ ] T41.1 Show five formulas/assumptions (including W0=1), build/display/save exact shared summary and three-curve figure.",
                "- [ ] T41.1 Show five formulas/assumptions (including W0=1) with scholarly citations for externally sourced performance-metric definitions, build/display/save exact shared summary and three-curve figure, and maintain source notes from `reports/references.bib`.",
                "T41.1",
            ),
            (
                "- [ ] AC41.1 (`T41.1`) Summary/figure/equations exactly match canonical shared calculations/presentation.",
                "- [ ] AC41.1 (`T41.1`) Summary/figure/equations exactly match canonical shared calculations/presentation and every external metric definition has a resolved scholarly citation.",
                "AC41.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        42,
        [
            (
                "- [ ] T42.2 Add final practical takeaway/limitations/future causal validation, refresh consulted MLA citations, write exact Step5 manifest, execute/verify full Step1–5 notebook.",
                "- [ ] T42.2 Add final practical takeaway/limitations/future causal validation with scholarly support, refresh `reports/references.bib`, verify all MLA 9 citations/source notes, render the final cited-only Works Cited, write exact Step5 manifest, and execute/verify full Step1–5 notebook.",
                "T42.2",
            ),
            (
                "- [ ] AC42.2 (`T42.2`) Takeaway/citations/manifest/hash/path coverage are exact and final notebook/artifacts fully consistent.",
                "- [ ] AC42.2 (`T42.2`) Takeaway/citations/source notes/Works Cited have complete verified scholarly provenance with no unresolved/orphan entries; manifest/hash/path coverage is exact and final notebook/artifacts are fully consistent.",
                "AC42.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        43,
        [
            (
                "- [ ] T43.2 Extend tests/checker for exact Step5 parity, required paths/assumptions, missing/stale failures, idempotence; regenerate README.",
                "- [ ] T43.2 Extend tests/checker for exact Step5 parity, required paths/assumptions, `reports/references.bib`/MLA 9 citation policy, missing/stale failures, and idempotence; regenerate README.",
                "T43.2",
            ),
        ],
    )

    text = update_pr_section(
        text,
        44,
        [
            (
                "- [ ] T44.1 Extend canonical readers/validation for Step5 benchmark comparison, summary, sensitivity, recommendation, limitations; no recomputation.",
                "- [ ] T44.1 Extend canonical readers/validation for Step5 benchmark comparison, summary, sensitivity, recommendation, limitations, and `reports/references.bib`; preserve resolved MLA 9 in-text citations/source notes/Works Cited with no recomputation.",
                "T44.1",
            ),
            (
                "- [ ] T44.2 Add tests for Step5 stale/missing parity while preserving nontechnical/code/model-name/template/name rules.",
                "- [ ] T44.2 Add tests for Step5 stale/missing parity and final citation integrity while preserving nontechnical/code/model-name/template/name rules outside bibliographic metadata.",
                "T44.2",
            ),
            (
                "- [ ] AC44.1 (`T44.1`) Builder requires exact Step5 decision content from canonical artifacts only.",
                "- [ ] AC44.1 (`T44.1`) Builder requires exact Step5 decision content from canonical artifacts only and a valid canonical scientific-source registry with resolved rendered citations.",
                "AC44.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        45,
        [
            (
                "- [ ] T45.1 Extend nontechnical report through Step5 with canonical metrics/figure/sensitivity, both-benchmark conclusion, recommendation, factors, costs, full-sample/non-OOS limitations, consulted MLA citations; correct cover/no page2.",
                "- [ ] T45.1 Extend nontechnical report through Step5 with canonical metrics/figure/sensitivity, both-benchmark conclusion, recommendation, factors, costs, full-sample/non-OOS limitations, MLA 9 in-text scholarly citations, official-primary source notes, and final Works Cited from `reports/references.bib`; correct cover/no page2.",
                "T45.1",
            ),
            (
                "- [ ] AC45.1 (`T45.1`) Final PDF has decision parity, nontechnical wording, correct citations/cover/limitations.",
                "- [ ] AC45.1 (`T45.1`) Final PDF has decision parity, nontechnical narrative wording, correct cover/limitations, resolved scholarly citations/source notes, and cited-only Works Cited entries; bibliographic titles are exempt from narrative technical-term checks.",
                "AC45.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        46,
        [
            (
                "- [ ] T46.1 Run existing exporter on final executed notebook; verify Step5 daily returns/summary/figure/sensitivity/interpretation and exact notebook hash.",
                "- [ ] T46.1 Run existing exporter on final executed notebook; verify Step5 daily returns/summary/figure/sensitivity/interpretation, MLA 9 citations/source notes/final Works Cited from `reports/references.bib`, and exact notebook hash.",
                "T46.1",
            ),
            (
                "- [ ] AC46.1 (`T46.1`) Final HTML is non-empty, contains all Step1–5 stored outputs, and hash equals canonical notebook without re-execution.",
                "- [ ] AC46.1 (`T46.1`) Final HTML is non-empty, contains all Step1–5 stored outputs plus citations/source notes/Works Cited, and hash equals canonical notebook without re-execution.",
                "AC46.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        47,
        [
            (
                "- [ ] T47.1 Extend checker/tests to Step5 manifest/artifacts, notebook/README technical parity, current HTML, and PDF decision parity; include state provenance.",
                "- [ ] T47.1 Extend checker/tests to Step5 manifest/artifacts, notebook/README technical parity, current HTML, PDF decision parity/state provenance, and final citation integrity against `reports/references.bib` for both notebook and PDF.",
                "T47.1",
            ),
            (
                "- [ ] AC47.1 (`T47.1`) Every Step5 stale/missing/hash/value sidecar defect fails at correct parity level.",
                "- [ ] AC47.1 (`T47.1`) Every Step5 stale/missing/hash/value sidecar defect and every unresolved/duplicate/orphan/URL-only citation or missing Works Cited/source-note defect fails at the correct parity level.",
                "AC47.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        48,
        [
            (
                "- [ ] T48.1 Implement exact allowlist/exclusions, non-symlink/path safety, deterministic sorted normalized ZIP, separate-PDF requirement, exact member/ZIP/PDF hashes and no-timestamp manifest, post-build reinspection.",
                "- [ ] T48.1 Implement exact allowlist including `reports/references.bib`, exclusions, non-symlink/path safety, deterministic sorted normalized ZIP, separate-PDF requirement, exact member/ZIP/PDF hashes and no-timestamp manifest, post-build reinspection; reject a missing/empty scientific citation registry.",
                "T48.1",
            ),
            (
                "- [ ] AC48.1 (`T48.1`) Builder output/manifest exactly satisfy fixed submission contract.",
                "- [ ] AC48.1 (`T48.1`) Builder output/manifest exactly satisfy fixed submission contract and include the canonical non-empty scientific citation registry.",
                "AC48.1",
            ),
        ],
    )

    text = update_pr_section(
        text,
        49,
        [
            (
                "- [ ] T49.1 Generate final ZIP/manifest from post-PR47 canonical files; inspect exact members/byte parity/forbidden exclusions and separately hashed PDF.",
                "- [ ] T49.1 Generate final ZIP/manifest from post-PR47 canonical files; inspect exact members including `reports/references.bib`, citation-registry byte parity, forbidden exclusions, and separately hashed PDF.",
                "T49.1",
            ),
            (
                "- [ ] T49.2 Update README/checker with exact ZIP path, separate PDF path, contents/rebuild/upload instructions; run backlog/README/sidecar/full quality suite.",
                "- [ ] T49.2 Update README/checker with exact ZIP path, separate PDF path, citation-registry contents/rebuild/upload instructions, and notebook/PDF scholarly-source requirements; run backlog/README/sidecar/full quality suite.",
                "T49.2",
            ),
            (
                "- [ ] AC49.1 (`T49.1`) ZIP/manifest/member hashes and separate-PDF exclusion exactly match fixed contract.",
                "- [ ] AC49.1 (`T49.1`) ZIP/manifest/member hashes, included canonical citation registry, and separate-PDF exclusion exactly match fixed contract.",
                "AC49.1",
            ),
        ],
    )

    text = replace_once(
        text,
        "- [ ] Final notebook fully executed; README exact technical parity; HTML exact duplicate; separate PDF nontechnical decision parity with template page1 only and visual QA.\n",
        "- [ ] Final notebook fully executed with resolved MLA 9 scholarly citations/source notes/final Works Cited; README exact technical parity; HTML exact duplicate; separate PDF nontechnical decision parity with resolved scholarly citations/source notes/Works Cited, template page1 only, and visual QA.\n",
        "final DoD citation requirement",
    )
    text = replace_once(
        text,
        "- [ ] Deterministic final ZIP contains exact executable allowlist and excludes standalone PDF/forbidden files; submission manifest hashes exact final bytes.\n",
        "- [ ] Deterministic final ZIP contains exact executable allowlist including `reports/references.bib` and excludes standalone PDF/forbidden files; submission manifest hashes exact final bytes.\n",
        "final DoD reference registry",
    )

    BACKLOG.write_text(text, encoding="utf-8")


def update_readme() -> None:
    text = README.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "`BACKLOG.md` is the **single canonical planning source**. It fixes PR dependencies, file ownership, public interfaces, schemas, numerical conventions, tie rules, test evidence, notebook serialization, sidecar parity, Step 5 backtesting semantics, final submission packaging, and the Git branch/status/commit contract for every PR.\n",
        "`BACKLOG.md` is the **single canonical planning source**. It fixes PR dependencies, file ownership, public interfaces, schemas, numerical conventions, tie rules, test evidence, notebook serialization, sidecar parity, Step 5 backtesting semantics, scientific-citation integrity, final submission packaging, and the Git branch/status/commit contract for every PR.\n",
        "README backlog summary",
    )

    policy = """## Scientific citation policy

The technical notebook and standalone PDF report must both contain **verifiable scientific source attribution**. `reports/references.bib` is the canonical bibliography registry and is created in PR-05, then maintained only by serialized notebook PRs when a new source is required.

The required citation standard is **MLA 9**: in-text citations are placed adjacent to externally sourced definitions, equations, methodological claims, and interpretations, and each artifact ends with a **Works Cited** section. Peer-reviewed papers and scholarly books/textbooks provide the academic support for Markov chains, HMM/EM/decoding, information criteria, performance metrics, and backtesting limitations. Official primary sources may additionally document Yahoo/Cboe/index/data definitions, but a bare URL or data-provider page does not substitute for scholarly support of theory or methodology.

Every notebook/PDF citation must resolve to `reports/references.bib`; every entry rendered in an artifact's Works Cited must be cited in that artifact. Duplicate keys, invented metadata, unresolved citations, bibliography-only orphan entries, and URL-only pseudo-citations are invalid. Figures and tables include concise source notes distinguishing the team's own calculations from external data or methodology.

The standalone PDF remains non-technical in its narrative. Bibliographic titles may naturally contain technical terminology; the no-model/no-algorithm wording rule applies to report narrative, not to Works Cited metadata. PR-31 introduces Step1–4 citation-integrity checks and PR-47 extends them through the final Step1–5 notebook, HTML, and PDF.

"""
    text = replace_once(
        text,
        "## Assignment implementation plan\n",
        policy + "## Assignment implementation plan\n",
        "README scientific citation section",
    )

    text = replace_once(
        text,
        "Standalone no-code report:\n\n```text\nreports/Stochastic_Modeling_GWP2_Report.pdf\n```\n\nThe PDF uses page 1 of `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf` as its cover and excludes the template instruction page. It is non-technical: decision results, recommended action, portfolio-impact factors, limitations, and practical takeaways without model/algorithm/library prose.\n\nParity policy:\n\n```text\nNotebook <-> README: exact technical-result parity\nNotebook <-> HTML: exact executed-notebook duplicate\nNotebook <-> standalone PDF: decision-result parity with non-technical wording\n```\n",
        "Standalone no-code report:\n\n```text\nreports/Stochastic_Modeling_GWP2_Report.pdf\n```\n\nCanonical scientific-source registry:\n\n```text\nreports/references.bib\n```\n\nThe PDF uses page 1 of `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf` as its cover and excludes the template instruction page. It is non-technical: decision results, recommended action, portfolio-impact factors, limitations, and practical takeaways without model/algorithm/library prose. It nevertheless includes MLA 9 in-text scholarly citations, source notes, and a final Works Cited derived from the canonical registry.\n\nParity policy:\n\n```text\nNotebook <-> README: exact technical-result parity\nNotebook <-> HTML: exact executed-notebook duplicate\nNotebook <-> standalone PDF: decision-result parity with non-technical wording\nNotebook/PDF citations -> reports/references.bib: resolved citation and Works-Cited integrity\n```\n",
        "README canonical citation artifacts",
    )

    text = replace_once(
        text,
        "The ZIP contains the notebook, HTML duplicate, README, `pyproject.toml`, Step 1 processed data, and the local `src/vix_regime_allocation` Python package needed to keep the notebook executable. The standalone PDF is explicitly excluded from the ZIP and remains a separate upload. The bundle is deterministic and hash-manifested.\n",
        "The ZIP contains the notebook, HTML duplicate, README, `pyproject.toml`, canonical `reports/references.bib`, Step 1 processed data, and the local `src/vix_regime_allocation` Python package needed to keep the notebook executable. The standalone PDF is explicitly excluded from the ZIP and remains a separate upload. The bundle is deterministic and hash-manifested.\n",
        "README final bundle citation registry",
    )

    README.write_text(text, encoding="utf-8")


def update_backlog_checker() -> None:
    text = BACKLOG_CHECKER.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "MAX_TASKS_PER_PR = 3\n",
        "MAX_TASKS_PER_PR = 3\nREFERENCE_OWNER_PRS = {5, 21, 22, 23, 24, 25, 40, 41, 42}\nCITATION_REQUIRED_PRS = {5, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}\n\nREQUIRED_CITATION_CONTRACT_FRAGMENTS = (\n    \"## Scientific citation contract\",\n    \"reports/references.bib\",\n    \"MLA 9\",\n    \"Peer-reviewed\",\n    \"Works Cited\",\n    \"No citation metadata may be invented\",\n)\n",
        "backlog checker citation constants",
    )

    text = replace_once(
        text,
        "    text = BACKLOG.read_text(encoding=\"utf-8\")\n    matches = list(PR_HEADER_RE.finditer(text))\n",
        "    text = BACKLOG.read_text(encoding=\"utf-8\")\n    missing_citation_contract = [\n        fragment for fragment in REQUIRED_CITATION_CONTRACT_FRAGMENTS if fragment not in text\n    ]\n    if missing_citation_contract:\n        _fail(\n            \"BACKLOG scientific citation contract is incomplete: \"\n            + \", \".join(missing_citation_contract)\n        )\n\n    matches = list(PR_HEADER_RE.finditer(text))\n",
        "backlog checker global citation validation",
    )

    text = replace_once(
        text,
        "        for path in files:\n            if path.startswith(\"/\") or \"..\" in Path(path).parts:\n                _fail(f\"PR-{pr_code} has non-repository-relative owned path {path!r}.\")\n\n        tasks = TASK_RE.findall(section)\n",
        "        for path in files:\n            if path.startswith(\"/\") or \"..\" in Path(path).parts:\n                _fail(f\"PR-{pr_code} has non-repository-relative owned path {path!r}.\")\n\n        if pr_number in REFERENCE_OWNER_PRS and \"reports/references.bib\" not in files:\n            _fail(f\"PR-{pr_code} must own reports/references.bib for serialized citation updates.\")\n        if pr_number in CITATION_REQUIRED_PRS:\n            lowered_section = section.lower()\n            if \"reports/references.bib\" not in section:\n                _fail(f\"PR-{pr_code} must reference the canonical scientific citation registry.\")\n            if \"citation\" not in lowered_section and \"works cited\" not in lowered_section:\n                _fail(f\"PR-{pr_code} must explicitly specify citation or Works Cited work.\")\n\n        tasks = TASK_RE.findall(section)\n",
        "backlog checker per-PR citation validation",
    )

    text = replace_once(
        text,
        "        \"deterministic Git branch/status/commit metadata, bounded atomicity, explicit file \"\n        \"ownership, and no forbidden ambiguous phrases.\"\n",
        "        \"deterministic Git branch/status/commit metadata, bounded atomicity, explicit file \"\n        \"ownership, scientific citation contracts, and no forbidden ambiguous phrases.\"\n",
        "backlog checker success message",
    )

    BACKLOG_CHECKER.write_text(text, encoding="utf-8")


def update_readme_checker() -> None:
    text = README_CHECKER.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '        "reports/Stochastic_Modeling_GWP2_Report.pdf",\n',
        '        "reports/Stochastic_Modeling_GWP2_Report.pdf",\n        "Scientific citation policy",\n        "reports/references.bib",\n        "MLA 9",\n        "Works Cited",\n        "Peer-reviewed papers",\n        "Notebook/PDF citations -> reports/references.bib",\n',
        "README checker scientific citation fragments",
    )
    README_CHECKER.write_text(text, encoding="utf-8")


def main() -> None:
    update_backlog()
    update_readme()
    update_backlog_checker()
    update_readme_checker()
    print("Scientific citation contract applied to backlog PRs and README/checkers.")


if __name__ == "__main__":
    main()
