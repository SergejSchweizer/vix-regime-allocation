import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "quality-gates.yml"


def main() -> None:
    readme_text = README.read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    threshold = pyproject["tool"]["coverage"]["report"]["fail_under"]
    if threshold != 90:
        raise SystemExit(f"Coverage threshold must remain 90, found {threshold!r}.")

    required_readme_fragments = (
        "90%",
        "ruff check .",
        "ruff format --check .",
        "mypy src",
        "unit-tests",
        "integration-tests",
        "coverage",
        "quality-gate",
        "README technical sidecar policy",
        "BACKLOG.md",
        "BACKLOG_STEPS_2_4.md",
        "notebooks/gwp2_vix_regime_allocation.ipynb",
        "reports/gwp2_vix_regime_allocation.html",
        "reports/Stochastic_Modeling_GWP2_Report.pdf",
        "reports/Template_Stochastic_Modeling_Group_Work_Project.pdf",
        "reports/generated/steps_2_4_manifest.json",
        "reports/generated/step3_selected_model.json",
        "analysis-sidecars",
        "Notebook <-> README: exact technical-result parity",
        "Notebook <-> HTML: exact executed-notebook duplicate",
        "Notebook <-> standalone PDF: decision-result parity",
    )
    missing_fragments = [
        fragment for fragment in required_readme_fragments if fragment not in readme_text
    ]
    if missing_fragments:
        raise SystemExit(
            "README sidecar is missing required contract text: " + ", ".join(missing_fragments)
        )

    required_jobs = (
        "lint",
        "type-check",
        "unit-tests",
        "integration-tests",
        "coverage",
        "readme-sidecar",
        "quality-gate",
    )
    for job in required_jobs:
        if re.search(rf"^  {re.escape(job)}:\s*$", workflow_text, flags=re.MULTILINE) is None:
            raise SystemExit(f"Workflow is missing required job: {job}")

    if "fail-under=90" not in workflow_text:
        raise SystemExit("Workflow must enforce coverage with --fail-under=90.")

    print("README sidecar contract is consistent with repository quality configuration.")


if __name__ == "__main__":
    main()
