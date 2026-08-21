from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "BACKLOG.md"
EXPECTED_FIRST_PR = 50
EXPECTED_LAST_PR = 68
MAX_TASKS_PER_PR = 3
REFERENCE_OWNER_PRS = {61}
CITATION_REQUIRED_PRS = {61, 62, 63, 66, 68}

REQUIRED_CITATION_CONTRACT_FRAGMENTS = (
    "reports/references.bib",
    "MLA 9",
    "Works Cited",
    "scholarly",
    "No bibliographic metadata is invented",
)

PR_HEADER_RE = re.compile(r"^#{2,3}\s+PR-(\d{2})\s+—\s+(.+)$", re.MULTILINE)
TASK_RE = re.compile(r"^- \[ \] T(\d{2})\.(\d+)\s+(.+)$", re.MULTILINE)
AC_RE = re.compile(
    r"^- \[ \] AC(\d{2})\.(\d+)\s+\(`T(\d{2})\.(\d+)`\)\s+(.+)$",
    re.MULTILINE,
)
DEPENDENCY_RE = re.compile(r"^\*\*Dependencies:\*\*\s*(.+)$", re.MULTILINE)
FILES_RE = re.compile(r"\*\*Files owned:\*\*\s*\n\s*```text\n(.*?)\n```", re.DOTALL)
GIT_BRANCH_RE = re.compile(r"^\*\*Git branch:\*\*\s+`([^`]+)`$", re.MULTILINE)
GIT_STATUS_RE = re.compile(
    r"^\*\*Git status:\*\*\s+`git status --short --branch` must show `([^`]+)` "
    r"and no staged, modified, or untracked files immediately before commit and merge\.$",
    re.MULTILINE,
)
COMMIT_MESSAGE_RE = re.compile(r"^\*\*Commit message:\*\*\s+`([^`]+)`$", re.MULTILINE)

FORBIDDEN_AMBIGUOUS_PHRASES = (
    "as needed",
    "relevant files",
    "etc.",
    "and so on",
    "tbd",
    "todo",
    "appropriate handling",
    "similar logic",
)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _branch_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main() -> None:
    text = BACKLOG.read_text(encoding="utf-8")
    missing_citation_contract = [
        fragment for fragment in REQUIRED_CITATION_CONTRACT_FRAGMENTS if fragment not in text
    ]
    if missing_citation_contract:
        _fail(
            "BACKLOG scientific citation contract is incomplete: "
            + ", ".join(missing_citation_contract)
        )

    matches = list(PR_HEADER_RE.finditer(text))
    numbers = [int(match.group(1)) for match in matches]
    expected = list(range(EXPECTED_FIRST_PR, EXPECTED_LAST_PR + 1))
    if numbers != expected:
        _fail(f"PR sequence must be exactly {expected}; found {numbers}.")

    all_task_ids: set[str] = set()
    all_ac_ids: set[str] = set()

    for position, match in enumerate(matches):
        pr_number = int(match.group(1))
        pr_code = f"{pr_number:02d}"
        title = match.group(2).strip()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        section = text[match.start() : end]

        for marker in (
            "**Agent lane:**",
            "**Dependencies:**",
            "**Git branch:**",
            "**Git status:**",
            "**Commit message:**",
            "**Files owned:**",
            "### Tasks",
            "### Acceptance criteria",
        ):
            if section.count(marker) != 1:
                _fail(f"PR-{pr_code} must contain exactly one {marker!r} marker.")

        dependency_match = DEPENDENCY_RE.search(section)
        if dependency_match is None:
            _fail(f"PR-{pr_code} has no parseable Dependencies line.")
        dependency_text = dependency_match.group(1).strip()
        dependency_refs = [int(value) for value in re.findall(r"PR-(\d{2})", dependency_text)]
        declares_none = dependency_text.lower() == "none"
        if not dependency_refs and not declares_none:
            _fail(
                f"PR-{pr_code} dependencies must name explicit earlier PRs or be exactly 'none'; "
                f"found {dependency_text!r}."
            )
        if declares_none and dependency_refs:
            _fail(f"PR-{pr_code} dependencies cannot mix 'none' with PR references.")
        if any(ref >= pr_number for ref in dependency_refs):
            _fail(f"PR-{pr_code} has a self/forward dependency: {dependency_text!r}.")
        if len(dependency_refs) != len(set(dependency_refs)):
            _fail(f"PR-{pr_code} repeats a dependency: {dependency_text!r}.")

        expected_branch = f"pr-{pr_code}-{_branch_slug(title)}"
        git_branch_match = GIT_BRANCH_RE.search(section)
        if git_branch_match is None or git_branch_match.group(1) != expected_branch:
            found_branch = git_branch_match.group(1) if git_branch_match else None
            _fail(
                f"PR-{pr_code} Git branch must be exactly {expected_branch!r}; "
                f"found {found_branch!r}."
            )

        git_status_match = GIT_STATUS_RE.search(section)
        if git_status_match is None or git_status_match.group(1) != expected_branch:
            _fail(
                f"PR-{pr_code} Git status must require a clean {expected_branch!r} branch "
                "using 'git status --short --branch'."
            )

        commit_message_match = COMMIT_MESSAGE_RE.search(section)
        expected_commit_message = f"PR-{pr_code} — {title}"
        if commit_message_match is None or commit_message_match.group(1) != expected_commit_message:
            _fail(f"PR-{pr_code} Commit message must be exactly {expected_commit_message!r}.")

        files_match = FILES_RE.search(section)
        if files_match is None:
            _fail(f"PR-{pr_code} Files owned must be one fenced text block.")
        files = [line.strip() for line in files_match.group(1).splitlines() if line.strip()]
        if not files:
            _fail(f"PR-{pr_code} Files owned is empty.")
        if len(files) != len(set(files)):
            _fail(f"PR-{pr_code} Files owned contains duplicate paths.")
        for path in files:
            if path.startswith("/") or ".." in Path(path).parts:
                _fail(f"PR-{pr_code} has non-repository-relative owned path {path!r}.")

        if pr_number in REFERENCE_OWNER_PRS and "reports/references.bib" not in files:
            _fail(f"PR-{pr_code} must own reports/references.bib for serialized citation updates.")
        if pr_number in CITATION_REQUIRED_PRS:
            lowered_section = section.lower()
            if "reports/references.bib" not in section:
                _fail(f"PR-{pr_code} must reference the canonical scientific citation registry.")
            if (
                "citation" not in lowered_section
                and "citing" not in lowered_section
                and "works cited" not in lowered_section
            ):
                _fail(f"PR-{pr_code} must explicitly specify citation or Works Cited work.")

        tasks = TASK_RE.findall(section)
        acceptances = AC_RE.findall(section)
        if not tasks or not acceptances:
            _fail(f"PR-{pr_code} must contain tasks and acceptance criteria.")
        if len(tasks) > MAX_TASKS_PER_PR:
            _fail(
                f"PR-{pr_code} has {len(tasks)} tasks; maximum is {MAX_TASKS_PER_PR} "
                "to preserve weak-agent atomicity."
            )

        task_suffixes: list[int] = []
        for task_pr, suffix, task_text in tasks:
            if task_pr != pr_code:
                _fail(f"PR-{pr_code} contains task T{task_pr}.{suffix} from another PR.")
            task_id = f"T{task_pr}.{suffix}"
            if task_id in all_task_ids:
                _fail(f"Duplicate task ID {task_id}.")
            all_task_ids.add(task_id)
            task_suffixes.append(int(suffix))
            lowered = task_text.lower()
            for phrase in FORBIDDEN_AMBIGUOUS_PHRASES:
                if phrase in lowered:
                    _fail(f"{task_id} contains ambiguous phrase {phrase!r}.")

        ac_suffixes: list[int] = []
        for ac_pr, ac_suffix, linked_pr, linked_suffix, ac_text in acceptances:
            if ac_pr != pr_code:
                _fail(f"PR-{pr_code} contains AC{ac_pr}.{ac_suffix} from another PR.")
            if (ac_pr, ac_suffix) != (linked_pr, linked_suffix):
                _fail(
                    f"AC{ac_pr}.{ac_suffix} must link to T{ac_pr}.{ac_suffix}; "
                    f"found T{linked_pr}.{linked_suffix}."
                )
            ac_id = f"AC{ac_pr}.{ac_suffix}"
            if ac_id in all_ac_ids:
                _fail(f"Duplicate acceptance ID {ac_id}.")
            all_ac_ids.add(ac_id)
            ac_suffixes.append(int(ac_suffix))
            lowered = ac_text.lower()
            for phrase in FORBIDDEN_AMBIGUOUS_PHRASES:
                if phrase in lowered:
                    _fail(f"{ac_id} contains ambiguous phrase {phrase!r}.")

        expected_suffixes = list(range(1, len(tasks) + 1))
        if task_suffixes != expected_suffixes:
            _fail(
                f"PR-{pr_code} task suffixes must be contiguous/ordered {expected_suffixes}; "
                f"found {task_suffixes}."
            )
        if ac_suffixes != expected_suffixes:
            _fail(
                f"PR-{pr_code} acceptance suffixes must exactly match tasks {expected_suffixes}; "
                f"found {ac_suffixes}."
            )

    if all_task_ids != {identifier.replace("AC", "T", 1) for identifier in all_ac_ids}:
        _fail("Global task/acceptance ID sets differ.")

    print(
        f"BACKLOG contract valid: PR-{EXPECTED_FIRST_PR:02d}..PR-{EXPECTED_LAST_PR:02d}, "
        f"{len(all_task_ids)} tasks, one-to-one acceptance coverage, explicit backward-only "
        "dependencies, deterministic Git branch/status/commit metadata, bounded atomicity, "
        "explicit file ownership, scientific citation contracts, and no forbidden ambiguous "
        "phrases."
    )


if __name__ == "__main__":
    main()
