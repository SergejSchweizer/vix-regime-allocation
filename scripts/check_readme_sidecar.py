import csv
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "quality-gates.yml"
AUTO_COMPLETE_WORKFLOW = ROOT / ".github" / "workflows" / "auto-complete.yml"
PDF_SIDECAR_WORKFLOW = ROOT / ".github" / "workflows" / "report-sync.yml"
SELECTED_MODEL = ROOT / "reports" / "generated" / "step3_selected_model.json"
MODEL_COMPARISON = ROOT / "reports" / "tables" / "step3_model_comparison.csv"
ALLOCATION_100 = ROOT / "reports" / "tables" / "step4_allocation_100_keep.csv"
ALLOCATION_60_40 = ROOT / "reports" / "tables" / "step4_allocation_60_40_spread.csv"
PERFORMANCE = ROOT / "reports" / "tables" / "step5_performance_summary.csv"
SENSITIVITY = ROOT / "reports" / "tables" / "step5_state_count_sensitivity.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: str) -> str:
    return f"{float(value) * 100:.4f}%"


def _four(value: str) -> str:
    return f"{float(value):.4f}"


def _require_fragments(text: str, fragments: tuple[str, ...], label: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise SystemExit(f"{label} is missing required contract text: " + ", ".join(missing))


def _check_artifact_parity(readme_text: str) -> None:
    selected = json.loads(SELECTED_MODEL.read_text(encoding="utf-8"))
    if selected.get("family") != "hmm":
        raise SystemExit("Selected-model artifact must have family='hmm'.")
    if selected.get("selected_states_path") != "reports/tables/step3_selected_states.csv":
        raise SystemExit("Selected-model artifact has the wrong selected-states path.")

    n_states = int(selected["n_states"])
    if f"HMM K={n_states}" not in readme_text:
        raise SystemExit("README selected HMM state count is stale.")

    comparison = {int(row["n_states"]): row for row in _read_csv(MODEL_COMPARISON)}
    if sorted(comparison) != [2, 3] or any(row["family"] != "hmm" for row in comparison.values()):
        raise SystemExit("Model-comparison artifact must contain only HMM K=2/K=3.")
    for k in (2, 3):
        if comparison[k]["bic"] not in readme_text:
            raise SystemExit(f"README is missing the canonical HMM K={k} BIC.")
    if comparison[3]["min_viterbi_occupancy"] not in readme_text:
        raise SystemExit("README is missing the canonical HMM K=3 occupancy diagnostic.")

    allocations = (
        (ALLOCATION_100, "100_keep"),
        (ALLOCATION_60_40, "60_40_spread"),
    )
    for path, method in allocations:
        rows = _read_csv(path)
        if not rows or any(row["method"] != method for row in rows):
            raise SystemExit(f"Allocation artifact {path.name} has the wrong method contract.")
        for row in rows:
            table_row = (
                f"| {row['state']} | {row['rank_1_asset']} | {row['rank_2_asset']} | "
                f"{float(row['TLT_weight']):.2f} | {float(row['GLD_weight']):.2f} | "
                f"{float(row['SPY_weight']):.2f} |"
            )
            if table_row not in readme_text:
                raise SystemExit(f"README allocation row is stale for {method}, state {row['state']}.")

    performance_rows = _read_csv(PERFORMANCE)
    expected_portfolios = [
        "hmm_100_keep",
        "hmm_60_40_spread",
        "equal_weight_monthly",
        "spy_buy_hold",
    ]
    if [row["portfolio"] for row in performance_rows] != expected_portfolios:
        raise SystemExit("Performance artifact does not have the canonical four-portfolio order.")
    if len({row["observations"] for row in performance_rows}) != 1:
        raise SystemExit("Canonical performance portfolios do not share one observation count.")
    if performance_rows[0]["observations"] not in readme_text:
        raise SystemExit("README performance observation count is stale.")

    for row in performance_rows:
        table_row = (
            f"| `{row['portfolio']}` | {_pct(row['cumulative_return'])} | "
            f"{_pct(row['annualized_return'])} | {_pct(row['annualized_volatility'])} | "
            f"{_four(row['sharpe_ratio'])} | {_pct(row['max_drawdown'])} |"
        )
        if table_row not in readme_text:
            raise SystemExit(f"README performance row is stale for {row['portfolio']}.")

    sensitivity_rows = _read_csv(SENSITIVITY)
    expected_sensitivity = [
        ("hmm", "2", "100_keep"),
        ("hmm", "2", "60_40_spread"),
        ("hmm", "3", "100_keep"),
        ("hmm", "3", "60_40_spread"),
    ]
    actual_sensitivity = [
        (row["family"], row["n_states"], row["method"]) for row in sensitivity_rows
    ]
    if actual_sensitivity != expected_sensitivity:
        raise SystemExit("Sensitivity artifact does not have the canonical four HMM combinations.")
    if len({row["observations"] for row in sensitivity_rows}) != 1:
        raise SystemExit("Sensitivity rows do not share one common observation count.")

    for row in sensitivity_rows:
        table_row = (
            f"| {row['n_states']} | `{row['method']}` | {_pct(row['cumulative_return'])} | "
            f"{_four(row['sharpe_ratio'])} | {_pct(row['max_drawdown'])} |"
        )
        if table_row not in readme_text:
            raise SystemExit(
                "README sensitivity row is stale for "
                f"K={row['n_states']} {row['method']}."
            )


def main() -> None:
    readme_text = README.read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    auto_complete_text = AUTO_COMPLETE_WORKFLOW.read_text(encoding="utf-8")
    pdf_sidecar_text = PDF_SIDECAR_WORKFLOW.read_text(encoding="utf-8")

    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    threshold = pyproject["tool"]["coverage"]["report"]["fail_under"]
    if threshold != 90:
        raise SystemExit(f"Coverage threshold must remain 90, found {threshold!r}.")

    required_readme_fragments = (
        "BACKLOG.md",
        "PR-50 through PR-68",
        "scripts/check_backlog_contract.py",
        "Gaussian Hidden Markov Models (HMMs) only",
        "HMM K=2",
        "100% Keep",
        "60/40 Spread",
        "reports/tables/step4_allocation_100_keep.csv",
        "reports/tables/step4_allocation_60_40_spread.csv",
        "hmm_100_keep",
        "hmm_60_40_spread",
        "equal_weight_monthly",
        "spy_buy_hold",
        "reports/tables/step5_state_count_sensitivity.csv",
        "reports/generated/step5_manifest.json",
        "scripts/check_analysis_consistency.py",
        "scripts/check_artifact_provenance.py",
        "90%",
        "ruff check .",
        "ruff format --check src tests scripts",
        "mypy src",
        "quality-gate",
        "notebooks/gwp2_vix_regime_allocation.ipynb",
        "reports/gwp2_vix_regime_allocation.html",
        "reports/Stochastic_Modeling_GWP2_Report.pdf",
        "Scientific citation policy",
        "reports/references.bib",
        "MLA 9",
        "Works Cited",
        "Notebook/PDF citations -> reports/references.bib",
        "does not make the assignment backtest causal or out-of-sample",
        "full-sample backtest",
    )
    _require_fragments(readme_text, required_readme_fragments, "README")

    stale_readme_fragments = (
        "PR-01 through PR-49",
        "preferred specification remains **Markov",
        "Step 2 implementation | Complete: Markov",
        "reports/tables/step4_allocation_mapping.csv",
        "The optional 60/40 rule is not used",
        "Markov K=2",
        "Markov K=3",
    )
    stale = [fragment for fragment in stale_readme_fragments if fragment in readme_text]
    if stale:
        raise SystemExit("README contains stale pre-HMM-revision claims: " + ", ".join(stale))

    if "BACKLOG_STEPS_2_4.md" in readme_text:
        raise SystemExit("README must reference only the canonical BACKLOG.md backlog.")

    _check_artifact_parity(readme_text)

    required_jobs = (
        "lint",
        "type-check",
        "unit-tests",
        "integration-tests",
        "readme-sidecar",
        "backlog-contract",
        "repository-hygiene",
        "analysis-consistency",
        "artifact-provenance",
        "coverage",
        "quality-gate",
    )
    for job in required_jobs:
        if re.search(rf"^  {re.escape(job)}:\s*$", workflow_text, flags=re.MULTILINE) is None:
            raise SystemExit(f"Workflow is missing required job: {job}")

    required_workflow_commands = (
        "python scripts/check_backlog_contract.py",
        "python scripts/check_repository_hygiene.py",
        "python scripts/check_analysis_consistency.py",
        "python scripts/check_artifact_provenance.py",
        "coverage report --fail-under=90",
        "ruff format --check src tests scripts",
    )
    for command in required_workflow_commands:
        if command not in workflow_text:
            raise SystemExit(f"Workflow must execute/enforce: {command}")

    for dependency in ("repository-hygiene", "analysis-consistency", "artifact-provenance"):
        if f"- {dependency}" not in workflow_text:
            raise SystemExit(f"Aggregate quality gate must depend on {dependency}.")

    required_auto_complete_fragments = (
        "name: Auto Complete",
        "workflow_run:",
        "- Quality Gates",
        "pull-requests: write",
        "contents: write",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'pull_request'",
        "VERIFIED_HEAD_SHA",
        "VERIFIED_BASE_SHA",
        "gh pr update-branch",
        "gh pr merge",
        "--delete-branch",
    )
    _require_fragments(
        auto_complete_text,
        required_auto_complete_fragments,
        "Auto-complete workflow",
    )

    required_pdf_sidecar_fragments = (
        "name: Notebook PDF Sidecar Sync",
        "python scripts/rebuild_analysis_review.py",
        "--execute --inplace",
        "SOURCE_DIGEST_KEY",
        "--to html",
        "python scripts/build_pdf_report.py",
        "python scripts/check_artifact_provenance.py",
        "Sync executed notebook and report sidecars",
        "canonical-notebook-report",
    )
    _require_fragments(
        pdf_sidecar_text,
        required_pdf_sidecar_fragments,
        "Notebook/report sync workflow",
    )

    print(
        "README sidecar matches the canonical HMM-only assignment artifacts, dual allocation "
        "methods, four-portfolio Step 5 results, citation policy, and quality-gate chain."
    )


if __name__ == "__main__":
    main()
