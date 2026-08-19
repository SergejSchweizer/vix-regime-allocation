from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import pytest

import vix_regime_allocation.notebook_helpers as module
from vix_regime_allocation.transform import OUTPUT_COLUMNS

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z1xkAAAAASUVORK5CYII="
)


def _write_csv(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()

    step1 = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05"],
            "TLT": [100.0, 101.0],
            "GLD": [200.0, 202.0],
            "SPY": [300.0, 303.0],
            "VIX": [15.0, 16.0],
            "TLT_log_return": [0.01, 0.01],
            "GLD_log_return": [0.02, 0.01],
            "SPY_log_return": [0.03, 0.01],
            "VIX_change": [-1.0, 1.0],
        }
    )
    assert tuple(step1.columns[1:]) == OUTPUT_COLUMNS
    _write_csv(tmp_path, "data/processed/step1_data.csv", step1)

    for n_states in (2, 3):
        _write_csv(
            tmp_path,
            f"reports/tables/step2_markov_{n_states}_thresholds.csv",
            pd.DataFrame(
                {
                    "state": range(n_states),
                    "lower_bound": range(n_states),
                    "upper_bound": range(1, n_states + 1),
                }
            ),
        )
        _write_csv(
            tmp_path,
            f"reports/tables/step2_markov_{n_states}_transition.csv",
            pd.DataFrame({"from_state": range(n_states), "state_0": [1.0] * n_states}),
        )
        _write_csv(
            tmp_path,
            f"reports/tables/step2_markov_{n_states}_stationary.csv",
            pd.DataFrame(
                {
                    "state": range(n_states),
                    "stationary_probability": [1.0 / n_states] * n_states,
                }
            ),
        )
        _write_csv(
            tmp_path,
            f"reports/tables/step2_hmm_{n_states}_parameters.csv",
            pd.DataFrame({"state": range(n_states), "mean_vix_change": range(n_states)}),
        )
        _write_csv(
            tmp_path,
            f"reports/tables/step2_hmm_{n_states}_transition.csv",
            pd.DataFrame({"from_state": range(n_states), "state_0": [1.0] * n_states}),
        )

    _write_csv(
        tmp_path,
        "reports/tables/step3_model_comparison.csv",
        pd.DataFrame({"family": ["markov"], "n_states": [2], "bic": [1.0]}),
    )
    selected_path = tmp_path / "reports/generated/step3_selected_model.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps({"family": "markov", "n_states": 2}), encoding="utf-8")
    _write_csv(
        tmp_path,
        "reports/tables/step3_state_asset_statistics.csv",
        pd.DataFrame(
            {
                "state": [0],
                "asset": ["SPY"],
                "mean_log_return": [0.01],
                "std_log_return": [0.02],
                "observations": [2],
            }
        ),
    )
    _write_csv(
        tmp_path,
        "reports/tables/step4_allocation_mapping.csv",
        pd.DataFrame(
            {
                "state": [0],
                "selected_asset": ["SPY"],
                "selection_mean_log_return": [0.01],
                "TLT_weight": [0.0],
                "GLD_weight": [0.0],
                "SPY_weight": [1.0],
            }
        ),
    )
    _write_csv(
        tmp_path,
        "reports/tables/step5_performance_summary.csv",
        pd.DataFrame(
            {
                "portfolio": ["regime_rotation"],
                "cumulative_return": [0.1],
                "annualized_return": [0.05],
                "annualized_volatility": [0.2],
                "sharpe_ratio": [0.25],
                "max_drawdown": [-0.3],
                "observations": [100],
            }
        ),
    )

    for relative in (
        "reports/figures/step1_etf_log_returns.png",
        "reports/figures/step1_vix_change.png",
        "reports/figures/step2_markov_vix_states.png",
        "reports/figures/step2_hmm_vix_states.png",
        "reports/figures/step2_hmm_smoothed_probabilities.png",
        "reports/figures/step3_state_asset_statistics.png",
        "reports/figures/step5_cumulative_performance.png",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_PNG_1X1)
    return tmp_path


def test_find_repo_root_from_nested_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    nested = root / "notebooks" / "nested"
    nested.mkdir(parents=True)
    assert module.find_repo_root(nested) == root


def test_find_repo_root_fails_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        module.find_repo_root(tmp_path)


def test_all_presentation_functions_display_canonical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    monkeypatch.chdir(root)
    seen: list[object] = []
    monkeypatch.setattr(module, "display", lambda value: seen.append(value))

    module.show_step1_sample()
    module.show_step1_figures()
    module.show_step2_markov_results()
    module.show_step2_hmm_results()
    module.show_step3_model_selection()
    module.show_step3_state_statistics()
    module.show_step4_allocation()
    module.show_step5_backtest()
    module.show_step5_sensitivity()

    assert len(seen) >= 20
    assert any(isinstance(value, pd.DataFrame) for value in seen)


def test_sensitivity_displays_existing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    _write_csv(
        root,
        "reports/tables/step5_state_count_sensitivity.csv",
        pd.DataFrame({"n_states": [2, 3], "sharpe_ratio": [0.2, 0.3]}),
    )
    monkeypatch.chdir(root)
    seen: list[object] = []
    monkeypatch.setattr(module, "display", lambda value: seen.append(value))

    module.show_step5_sensitivity()

    frames = [value for value in seen if isinstance(value, pd.DataFrame)]
    assert frames
    assert list(frames[-1]["n_states"]) == [2, 3]


def test_step5_schema_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    _write_csv(root, "reports/tables/step5_performance_summary.csv", pd.DataFrame({"bad": [1]}))
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="canonical schema"):
        module.show_step5_backtest()


def test_step1_schema_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    frame = pd.read_csv(root / "data/processed/step1_data.csv").drop(columns="VIX_change")
    _write_csv(root, "data/processed/step1_data.csv", frame)
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="canonical schema"):
        module.show_step1_sample()
