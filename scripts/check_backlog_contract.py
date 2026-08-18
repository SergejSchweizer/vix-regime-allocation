from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "BACKLOG.md"
EXPECTED_FIRST_PR = 1
EXPECTED_LAST_PR = 49

PR_HEADER_RE = re.compile(r"^#{2,3}\s+PR-(\d{2})\s+—\s+.+$", re.MULTILINE)
TASK_RE = re.compile(r"^- \[ \] T(\d{2})\.(\d+)\s+", re.MULTILINE)
AC_RE = re.compile(r"^- \[ \] AC(\d{2})\.(\d+)\s+\(`T(\d{2})\.(\d+)`\)\s+", re.MULTILINE)
DEPENDENCY_RE = re.compile(r"^\*\*Dependencies:\*\*\s*(.+)$", re.MULTILINE)


def _fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    text = BACKLOG.read_text(encoding="utf-8")
    matches = list(PR_HEADER_RE.finditer(text))
    if not matches:
        _fail("BACKLOG.md contains no PR definitions.")

    numbers = [int(match.group(1)) for match in matches]
    expected = list(range(EXPECTED_FIRST_PR, EXPECTED_LAST_PR + 1))
    if numbers != expected:
        _fail(f"PR sequence must be exactly {expected}; found {numbers}.")

    all_task_ids: set[str] = set()
    all_ac_ids: set[str] = set()

    for position, match in enumerate(matches):
        pr_number = int(match.group(1))
        pr_code = f"{pr_number:02d}"
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        section = text[match.start() : end]

        for required_marker in (
            "**Agent lane:**",
            "**Dependencies:**",
            "**Files owned:**",
            "### Tasks",
            "### Acceptance criteria",
        ):
            if required_marker not in section:
                _fail(f"PR-{pr_code} is missing required marker {required_marker!r}.")

        dependency_match = DEPENDENCY_RE.search(section)
        if dependency_match is None:
            _fail(f"PR-{pr_code} has no parseable Dependencies line.")
        dependency_text = dependency_match.group(1).strip()
        dependency_refs = [int(value) for value in re.findall(r"PR-(\d{2})", dependency_text)]
        declares_none = bool(re.search(r"\bnone\b", dependency_text, flags=re.IGNORECASE))
        if not dependency_refs and not declares_none:
            _fail(
                f"PR-{pr_code} dependencies must name explicit earlier PRs or declare none; "
                f"found {dependency_text!r}."
            )
        if declares_none and dependency_refs:
            _fail(f"PR-{pr_code} dependencies cannot mix 'none' with PR references.")
        if any(ref >= pr_number for ref in dependency_refs):
            _fail(
                f"PR-{pr_code} has self/forward dependency in {dependency_text!r}; "
                "all dependencies must be lower-numbered PRs."
            )
        if len(dependency_refs) != len(set(dependency_refs)):
            _fail(f"PR-{pr_code} repeats a dependency: {dependency_text!r}.")

        tasks = TASK_RE.findall(section)
        acceptances = AC_RE.findall(section)
        if not tasks:
            _fail(f"PR-{pr_code} has no tasks.")
        if not acceptances:
            _fail(f"PR-{pr_code} has no acceptance criteria.")

        task_suffixes: list[int] = []
        for task_pr, suffix in tasks:
            if task_pr != pr_code:
                _fail(f"PR-{pr_code} contains task T{task_pr}.{suffix} from another PR.")
            task_id = f"T{task_pr}.{suffix}"
            if task_id in all_task_ids:
                _fail(f"Duplicate task ID {task_id}.")
            all_task_ids.add(task_id)
            task_suffixes.append(int(suffix))

        ac_suffixes: list[int] = []
        for ac_pr, ac_suffix, linked_pr, linked_suffix in acceptances:
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

        expected_suffixes = list(range(1, max(task_suffixes) + 1))
        if task_suffixes != expected_suffixes:
            _fail(
                f"PR-{pr_code} task suffixes must be contiguous and ordered {expected_suffixes}; "
                f"found {task_suffixes}."
            )
        if ac_suffixes != expected_suffixes:
            _fail(
                f"PR-{pr_code} acceptance suffixes must exactly match tasks {expected_suffixes}; "
                f"found {ac_suffixes}."
            )

    if len(all_task_ids) != len(all_ac_ids):
        _fail(
            f"Global task/acceptance cardinality mismatch: {len(all_task_ids)} tasks vs "
            f"{len(all_ac_ids)} acceptance criteria."
        )

    print(
        f"BACKLOG contract valid: {len(numbers)} PRs, {len(all_task_ids)} tasks, "
        "one-to-one acceptance coverage, explicit backward-only dependencies."
    )


if __name__ == "__main__":
    main()
