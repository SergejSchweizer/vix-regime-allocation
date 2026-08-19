from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    ".github/_tmp_",
    ".github/workflows/_tmp_",
    "scripts/_tmp_",
)


def find_forbidden_files(root: Path = ROOT) -> list[str]:
    """Return tracked-worktree-style temporary paths that must never ship."""
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(FORBIDDEN_PREFIXES):
            forbidden.append(relative)
    return sorted(forbidden)


def main() -> None:
    forbidden = find_forbidden_files()
    if forbidden:
        joined = "\n".join(f"- {path}" for path in forbidden)
        raise SystemExit(f"Forbidden temporary repository files detected:\n{joined}")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_notebook_orchestration.py")],
        cwd=ROOT,
        check=True,
    )
    print("Repository hygiene check passed.")


if __name__ == "__main__":
    main()
