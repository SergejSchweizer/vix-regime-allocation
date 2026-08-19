from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.allocation import build_state_allocation
from vix_regime_allocation.backtest import ROTATION_DETAIL_COLUMNS, build_rotation_returns
from vix_regime_allocation.backtest_plot import plot_cumulative_performance
from vix_regime_allocation.backtest_summary import (
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    build_comparison,
    build_performance_summary,
)
from vix_regime_allocation.benchmarks import (
    build_equal_weight_monthly_returns,
    build_spy_buy_hold_returns,
)
from vix_regime_allocation.sensitivity import SENSITIVITY_COLUMNS, build_state_count_sensitivity
from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.transform import OUTPUT_COLUMNS

pytestmark = pytest.mark.integration


def _fixture() -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    index = pd.DatetimeIndex(
        [
            "2026-01-27",
            "2026-01-28",
            "2026-01-29",
            "2026-01-30",
            "2026-02-02",
            "2026-02-03",
            "2026-02-04",
            "2026-02-05",
            "2026-02-06",
        ],
        name="Date",
    )
    frame = pd.DataFrame(index=index)
    frame["TLT"] = 100.0 + np.arange(len(index), dtype=float)
    frame["GLD"] = 200.0 + np.arange(len(index), dtype=float)
    frame["SPY"] = 300.0 + np.arange(len(index), dtype=float)
    frame["VIX"] = [20.0, 21.0, 19.0, 22.0, 18.0, 23.0, 17.0, 24.0, 20.0]
    simple = {
        "TLT": [0.0, 0.01, -0.01, 0.02, 0.01, -0.015, 0.025, -0.005, 0.015],
        "GLD": [0.0, 0.005, 0.01, -0.005, 0.015, 0.005, -0.01, 0.02, -0.005],
        "SPY": [0.0, 0.02, -0.015, 0.01, -0.02, 0.03, 0.005, -0.01, 0.025],
    }
    for asset in ("TLT", "GLD", "SPY"):
        frame[f"{asset}_log_return"] = np.log1p(simple[asset])
    frame["VIX_change"] = [0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -4.0]
    data = frame.loc[:, list(OUTPUT_COLUMNS)]
    states_by_k = {
        2: pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1], index=index, name="state", dtype=int),
        3: pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], index=index, name="state", dtype=int),
    }
    return data, states_by_k


def test_step5_source_pipeline_end_to_end_offline(tmp_path: Path) -> None:
    data, states_by_k = _fixture()
    selected_states = states_by_k[2]

    statistics = compute_state_asset_statistics(data, selected_states)
    allocation = build_state_allocation(statistics)
    rotation = build_rotation_returns(data, selected_states, allocation)

    assert tuple(rotation.columns) == ROTATION_DETAIL_COLUMNS
    assert rotation.index.equals(data.index[1:])
    assert rotation["decision_date"].tolist() == data.index[:-1].tolist()
    assert rotation["decision_state"].tolist() == selected_states.iloc[:-1].tolist()

    comparison_index = pd.DatetimeIndex(rotation.index, name="Date")
    equal_weight = build_equal_weight_monthly_returns(data, comparison_index)
    spy = build_spy_buy_hold_returns(data, comparison_index)
    comparison = build_comparison(rotation, equal_weight, spy)

    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert comparison.index.equals(comparison_index)
    assert len(comparison) == len(data) - 1
    expected_spy = np.expm1(data.loc[comparison_index, "SPY_log_return"])
    np.testing.assert_allclose(comparison["spy_buy_hold"], expected_spy)

    summary = build_performance_summary(comparison)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary["portfolio"].tolist() == list(COMPARISON_COLUMNS)
    assert summary["observations"].tolist() == [len(comparison)] * 3
    assert np.isfinite(summary.drop(columns=["portfolio"]).to_numpy(dtype=float)).all()

    figure_path = tmp_path / "step5_cumulative_performance.png"
    plot_cumulative_performance(comparison, figure_path)
    assert figure_path.is_file() and figure_path.stat().st_size > 0

    sensitivity = build_state_count_sensitivity(data, "markov", states_by_k)
    assert tuple(sensitivity.columns) == SENSITIVITY_COLUMNS
    assert sensitivity["family"].tolist() == ["markov", "markov"]
    assert sensitivity["n_states"].tolist() == [2, 3]
    assert sensitivity["observations"].nunique() == 1
    assert sensitivity["observations"].iloc[0] == len(data) - 1

    jan_mask = comparison.index.month == 1
    assert int(jan_mask.sum()) == 3
    assert int((comparison.index.month == 2).sum()) == 5
