from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.allocation import build_state_allocation
from vix_regime_allocation.backtest_plot import plot_four_portfolio_cumulative_performance
from vix_regime_allocation.backtest_summary import (
    SUMMARY_COLUMNS,
    build_four_portfolio_performance_summary,
)
from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.model_selection import (
    COMPARISON_COLUMNS,
    build_hmm_model_comparison,
    select_preferred_hmm,
)
from vix_regime_allocation.sensitivity import (
    HMM_SENSITIVITY_COLUMNS,
    build_hmm_state_count_sensitivity,
)
from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.strategy_comparison import (
    COMPARISON_COLUMNS as STEP5_COLUMNS,
    build_dual_method_comparison,
)
from vix_regime_allocation.transform import OUTPUT_COLUMNS

pytestmark = pytest.mark.integration


def _synthetic_step1() -> pd.DataFrame:
    block = np.repeat(np.array([0, 1, 2], dtype=int), 40)
    latent = np.tile(block, 6)
    observations = len(latent)
    index = pd.bdate_range("2022-01-03", periods=observations, name="Date")
    phase = np.arange(observations, dtype=float)

    vix_change = np.choose(latent, [-1.8, 0.2, 2.1]).astype(float)
    vix_change += 0.08 * np.sin(phase / 3.0) + 0.03 * np.cos(phase / 7.0)

    tlt = np.choose(latent, [0.0020, 0.0003, -0.0004]).astype(float)
    gld = np.choose(latent, [0.0007, 0.0016, 0.0005]).astype(float)
    spy = np.choose(latent, [-0.0005, 0.0008, 0.0022]).astype(float)
    tlt += 0.00005 * np.sin(phase / 5.0)
    gld += 0.00005 * np.cos(phase / 6.0)
    spy += 0.00005 * np.sin(phase / 8.0)

    frame = pd.DataFrame(
        {
            "TLT": 100.0 * np.exp(np.cumsum(tlt)),
            "GLD": 100.0 * np.exp(np.cumsum(gld)),
            "SPY": 100.0 * np.exp(np.cumsum(spy)),
            "VIX": 30.0 + np.cumsum(vix_change),
            "TLT_log_return": tlt,
            "GLD_log_return": gld,
            "SPY_log_return": spy,
            "VIX_change": vix_change,
        },
        index=index,
        columns=list(OUTPUT_COLUMNS),
    )
    return frame


def _fit_candidates(data: pd.DataFrame) -> list[dict[str, object]]:
    vix_change = data["VIX_change"].rename("VIX_change")
    return [evaluate_hmm_candidate(vix_change, n_states) for n_states in (2, 3)]


def _assert_method_weights(allocation: pd.DataFrame, method: str) -> None:
    weights = allocation[["TLT_weight", "GLD_weight", "SPY_weight"]].to_numpy(dtype=float)
    if method == "100_keep":
        expected = np.array([0.0, 0.0, 1.0])
    else:
        expected = np.array([0.0, 0.4, 0.6])
    np.testing.assert_allclose(np.sort(weights, axis=1), np.tile(expected, (len(weights), 1)))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_hmm_only_dual_allocation_pipeline_is_deterministic_and_lagged(tmp_path: Path) -> None:
    forbidden_before = {
        name for name in sys.modules if name.startswith("vix_regime_allocation.markov")
    }
    data = _synthetic_step1()
    candidates = _fit_candidates(data)
    comparison = build_hmm_model_comparison(candidates)

    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert comparison["family"].tolist() == ["hmm", "hmm"]
    assert comparison["n_states"].tolist() == [2, 3]
    assert comparison["valid"].any()

    selection = select_preferred_hmm(comparison, candidates)
    assert selection["family"] == "hmm"
    assert selection["n_states"] in (2, 3)
    states = selection["states"]
    assert isinstance(states, pd.Series)
    assert states.index.equals(data.index)
    assert states.name == "state"

    statistics = compute_state_asset_statistics(data, states)
    keep = build_state_allocation(statistics, "100_keep")
    spread = build_state_allocation(statistics, "60_40_spread")
    _assert_method_weights(keep, "100_keep")
    _assert_method_weights(spread, "60_40_spread")

    step5, rotations = build_dual_method_comparison(data, states, statistics)
    assert tuple(step5.columns) == STEP5_COLUMNS
    assert set(rotations) == {"100_keep", "60_40_spread"}
    assert rotations["100_keep"].index.equals(step5.index)
    assert rotations["60_40_spread"].index.equals(step5.index)

    expected_return_dates = data.index[1:]
    expected_decision_dates = data.index[:-1]
    expected_decision_states = states.iloc[:-1].to_numpy(dtype=int)
    assert step5.index.equals(expected_return_dates)
    for detail in rotations.values():
        assert detail["decision_date"].array.equals(expected_decision_dates.array)
        np.testing.assert_array_equal(
            detail["decision_state"].to_numpy(dtype=int), expected_decision_states
        )

    summary = build_four_portfolio_performance_summary(step5)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary["portfolio"].tolist() == list(STEP5_COLUMNS)
    assert summary["observations"].tolist() == [len(step5)] * 4

    figure_path = tmp_path / "step5_cumulative_performance.png"
    plot_four_portfolio_cumulative_performance(step5, figure_path)
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0

    states_by_k: dict[int, pd.Series] = {}
    for candidate in candidates:
        fit = candidate["fit"]
        assert isinstance(fit, HMMFitResult)
        states_by_k[fit.n_states] = fit.states
    sensitivity = build_hmm_state_count_sensitivity(data, states_by_k)
    assert tuple(sensitivity.columns) == HMM_SENSITIVITY_COLUMNS
    assert sensitivity[["family", "n_states", "method"]].to_records(index=False).tolist() == [
        ("hmm", 2, "100_keep"),
        ("hmm", 2, "60_40_spread"),
        ("hmm", 3, "100_keep"),
        ("hmm", 3, "60_40_spread"),
    ]
    assert sensitivity["observations"].tolist() == [len(data) - 1] * 4

    forbidden_after = {
        name for name in sys.modules if name.startswith("vix_regime_allocation.markov")
    }
    assert forbidden_after == forbidden_before
