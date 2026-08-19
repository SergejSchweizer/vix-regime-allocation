import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "quality-gates.yml"
AUTO_COMPLETE_WORKFLOW = ROOT / ".github" / "workflows" / "auto-complete.yml"
STEP5_SUMMARY = ROOT / "reports" / "tables" / "step5_performance_summary.csv"
STEP5_SENSITIVITY = ROOT / "reports" / "tables" / "step5_state_count_sensitivity.csv"
STEP5_MANIFEST = ROOT / "reports" / "generated" / "step5_manifest.json"


def main() -> None:
    readme_text = README.read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    auto_complete_text = AUTO_COMPLETE_WORKFLOW.read_text(encoding="utf-8")

    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    threshold = pyproject["tool"]["coverage"]["report"]["fail_under"]
    if threshold != 90:
        raise SystemExit(f"Coverage threshold must remain 90, found {threshold!r}.")

    required_readme_fragments = (
        "BACKLOG.md",
        "PR-01 through PR-49",
        "scripts/check_backlog_contract.py",
        "scripts/check_repository_hygiene.py",
        "Git workflow per backlog PR",
        "git status --short --branch",
        "PR-01 — Yahoo adjusted-close loader",
        "Step 1 implementation | Complete",
        "Step 5 computational implementation | Complete",
        "90%",
        "ruff check .",
        "ruff format --check src tests scripts",
        "mypy src",
        "unit-tests",
        "integration-tests",
        "repository-hygiene",
        "quality-gate",
        "Backlog contract",
        ".github/workflows/auto-complete.yml",
        "Auto Complete",
        "successful Quality Gates",
        "notebooks/gwp2_vix_regime_allocation.ipynb",
        "reports/gwp2_vix_regime_allocation.html",
        "reports/Stochastic_Modeling_GWP2_Report.pdf",
        "Scientific citation policy",
        "reports/references.bib",
        "MLA 9",
        "Works Cited",
        "Peer-reviewed papers",
        "Notebook/PDF citations -> reports/references.bib",
        "reports/Template_Stochastic_Modeling_Group_Work_Project.pdf",
        "reports/tables/step3_selected_states.csv",
        "reports/tables/step5_daily_returns.csv",
        "reports/tables/step5_performance_summary.csv",
        "reports/tables/step5_state_count_sensitivity.csv",
        "reports/figures/step5_cumulative_performance.png",
        "reports/generated/step5_manifest.json",
        "dist/MScFE_622_GWP2_submission.zip",
        "reports/generated/submission_manifest.json",
        "Notebook <-> README: exact technical-result parity",
        "Notebook <-> HTML: exact executed-notebook duplicate",
        "Notebook <-> standalone PDF: decision-result parity",
        "ΔVIX",
        "one-observation lag",
        "contemporaneous",
        "in-sample",
        "does **not** make this implementation causal or out-of-sample",
    )
    missing = [fragment for fragment in required_readme_fragments if fragment not in readme_text]
    if missing:
        raise SystemExit("README is missing required contract text: " + ", ".join(missing))

    if "BACKLOG_STEPS_2_4.md" in readme_text:
        raise SystemExit("README must reference only the canonical BACKLOG.md backlog.")

    if STEP5_SUMMARY.exists() and STEP5_SUMMARY.stat().st_size > 0:
        stale_markers = (
            "Step 5 implementation | Not started",
            "Step 5 computational implementation | Not started",
        )
        stale = [marker for marker in stale_markers if marker in readme_text]
        if stale:
            raise SystemExit("README has stale Step 5 status text: " + ", ".join(stale))

    for artifact in (STEP5_SUMMARY, STEP5_SENSITIVITY, STEP5_MANIFEST):
        if not artifact.exists() or artifact.stat().st_size == 0:
            raise SystemExit(f"Required Step 5 canonical artifact is missing or empty: {artifact}")

    required_jobs = (
        "lint",
        "type-check",
        "unit-tests",
        "integration-tests",
        "readme-sidecar",
        "backlog-contract",
        "repository-hygiene",
        "coverage",
        "quality-gate",
    )
    for job in required_jobs:
        if re.search(rf"^  {re.escape(job)}:\s*$", workflow_text, flags=re.MULTILINE) is None:
            raise SystemExit(f"Workflow is missing required job: {job}")

    required_workflow_commands = (
        "python scripts/check_backlog_contract.py",
        "python scripts/check_repository_hygiene.py",
        "coverage report --fail-under=90",
        "ruff format --check src tests scripts",
    )
    missing_commands = [
        command for command in required_workflow_commands if command not in workflow_text
    ]
    if missing_commands:
        raise SystemExit(
            "Workflow is missing required quality-gate commands: " + ", ".join(missing_commands)
        )

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
    missing_auto_complete = [
        fragment
        for fragment in required_auto_complete_fragments
        if fragment not in auto_complete_text
    ]
    if missing_auto_complete:
        raise SystemExit(
            "Auto-complete workflow is missing required safety contract text: "
            + ", ".join(missing_auto_complete)
        )

    print(
        "README sidecar is consistent with Step 1–5 artifacts, the canonical backlog, "
        "quality gates, repository hygiene, and the auto-complete workflow."
    )


if __name__ == "__main__":
    main()
