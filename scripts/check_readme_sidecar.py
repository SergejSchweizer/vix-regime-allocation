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
        "README sidecar policy",
        "BACKLOG.md",
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
