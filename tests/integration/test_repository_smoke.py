from pathlib import Path

import pytest

import vix_regime_allocation

pytestmark = pytest.mark.integration


def test_repository_contract_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]

    required_paths = (
        root / "README.md",
        root / "BACKLOG.md",
        root / "pyproject.toml",
        root / ".github" / "workflows" / "quality-gates.yml",
    )

    for path in required_paths:
        assert path.exists(), f"Missing required repository contract file: {path}"

    assert vix_regime_allocation.__version__ == "0.1.0"
