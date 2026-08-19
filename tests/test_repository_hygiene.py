from pathlib import Path

from scripts.check_repository_hygiene import find_forbidden_files


def test_hygiene_detects_temporary_files(tmp_path: Path) -> None:
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".github/workflows/_tmp_build.yml").write_text("x", encoding="utf-8")
    (tmp_path / "scripts/_tmp_builder.py").write_text("x", encoding="utf-8")
    (tmp_path / "scripts/real.py").write_text("x", encoding="utf-8")
    assert find_forbidden_files(tmp_path) == [
        ".github/workflows/_tmp_build.yml",
        "scripts/_tmp_builder.py",
    ]


def test_hygiene_accepts_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/check.py").write_text("ok", encoding="utf-8")
    assert find_forbidden_files(tmp_path) == []
