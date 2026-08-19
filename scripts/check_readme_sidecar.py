import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "quality-gates.yml"
AUTO_COMPLETE_WORKFLOW = ROOT / ".github" / "workflows" / "auto-complete.yml"


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
        "Git workflow per backlog PR",
        "git status --short --branch",
        "PR-01 — Yahoo adjusted-close loader",
        "Step 1 implementation | Complete",
        "90%",
        "ruff check .",
        "ruff format --check src tests scripts",
        "mypy src",
        "unit-tests",
        "integration-tests",
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
        "dist/MScFE_622_GWP2_submission.zip",
        "reports/generated/submission_manifest.json",
        "Notebook <-> README: exact technical-result parity",
        "Notebook <-> HTML: exact executed-notebook duplicate",
        "Notebook <-> standalone PDF: decision-result parity",
        "does **not** make this implementation causal or out-of-sample",
    )
    missing = [fragment for fragment in required_readme_fragments if fragment not in readme_text]
    if missing:
        raise SystemExit("README is missing required contract text: " + ", ".join(missing))

    if "BACKLOG_STEPS_2_4.md" in readme_text:
        raise SystemExit("README must reference only the canonical BACKLOG.md backlog.")

    required_jobs = (
        "lint",
        "type-check",
        "unit-tests",
        "integration-tests",
        "readme-sidecar",
        "backlog-contract",
        "coverage",
        "quality-gate",
    )
    for job in required_jobs:
        if re.search(rf"^  {re.escape(job)}:\s*$", workflow_text, flags=re.MULTILINE) is None:
            raise SystemExit(f"Workflow is missing required job: {job}")

    if "python scripts/check_backlog_contract.py" not in workflow_text:
        raise SystemExit("Workflow must execute scripts/check_backlog_contract.py.")
    if "coverage report --fail-under=90" not in workflow_text:
        raise SystemExit("Workflow must enforce coverage with --fail-under=90.")
    if "ruff format --check src tests scripts" not in workflow_text:
        raise SystemExit("Workflow must format-check the Python source/test/script trees.")

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
        "README planning sidecar is consistent with the canonical backlog, quality gates, "
        "and auto-complete workflow."
    )


if __name__ == "__main__":
    main()
