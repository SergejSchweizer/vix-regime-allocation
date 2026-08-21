from __future__ import annotations

import numpy as np
import pandas as pd

from vix_regime_allocation.state_statistics import STATISTICS_COLUMNS
from vix_regime_allocation.strategy_comparison import (
    COMPARISON_COLUMNS,
    build_dual_method_comparison,
)
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _data() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"], name="Date"
    )
    simple = {
        "TLT": [0.0, 0.10, 0.20, 0.30],
        "GLD": [0.0, 0.01, 0.02, 0.03],
        "SPY": [0.0, -0.05, 0.04, -0.02],
    }
    frame = pd.DataFrame(index=index)
    frame["TLT"] = [100.0, 101.0, 102.0, 103.0]
    frame["GLD"] = [200.0, 201.0, 202.0, 203.0]
    frame["SPY"] = [300.0, 301.0, 302.0, 303.0]
    frame["VIX"] = [20.0, 21.0, 19.0, 22.0]
    for asset in ("TLT", "GLD", "SPY"):
        frame[f"{asset}_log_return"] = np.log1p(simple[asset])
    frame["VIX_change"] = [0.0, 1.0, -2.0, 3.0]
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def _states(data: pd.DataFrame) -> pd.Series:
    return pd.Series([0, 1, 1, 0], index=data.index, name="state", dtype="int64")


def _statistics() -> pd.DataFrame:
    rows = [
        (0, "TLT", 0.001),
        (0, "GLD", 0.005),
        (0, "SPY", 0.010),
        (1, "TLT", 0.020),
        (1, "GLD", 0.010),
        (1, "SPY", 0.001),
    ]
    return pd.DataFrame(
        [
            {
                "state": state,
                "asset": asset,
                "mean_log_return": mean,
                "std_log_return": 0.01,
                "observations": 10,
            }
            for state, asset, mean in rows
        ],
        columns=list(STATISTICS_COLUMNS),
    )


def test_dual_method_comparison_has_exact_four_series_and_identical_dates() -> None:
    data = _data()
    comparison, rotations = build_dual_method_comparison(data, _states(data), _statistics())

    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert comparison.index.equals(data.index[1:])
    assert set(rotations) == {"100_keep", "60_40_spread"}
    assert rotations["100_keep"].index.equals(comparison.index)
    assert rotations["60_40_spread"].index.equals(comparison.index)


def test_previous_row_state_drives_both_hmm_methods_with_exact_arithmetic() -> None:
    data = _data()
    comparison, rotations = build_dual_method_comparison(data, _states(data), _statistics())

    # State 0 ranks SPY > GLD > TLT. State 1 ranks TLT > GLD > SPY.
    np.testing.assert_allclose(comparison["hmm_100_keep"], [-0.05, 0.20, 0.30])
    np.testing.assert_allclose(
        comparison["hmm_60_40_spread"],
        [0.6 * -0.05 + 0.4 * 0.01, 0.6 * 0.20 + 0.4 * 0.02, 0.6 * 0.30 + 0.4 * 0.03],
    )
    assert rotations["100_keep"]["decision_state"].tolist() == [0, 1, 1]
    assert rotations["60_40_spread"]["decision_state"].tolist() == [0, 1, 1]


def test_benchmarks_use_same_dates_and_existing_return_conventions() -> None:
    data = _data()
    comparison, _ = build_dual_method_comparison(data, _states(data), _statistics())

    # February starts at the first comparison row, so equal-weight resets to 1/3 there.
    first_equal_weight = (0.10 + 0.01 - 0.05) / 3.0
    assert comparison.iloc[0]["equal_weight_monthly"] == np.float64(first_equal_weight)
    np.testing.assert_allclose(comparison["spy_buy_hold"], [-0.05, 0.04, -0.02])


def test_state_change_impacts_both_strategies_only_on_following_return_row() -> None:
    data = _data()
    baseline = pd.Series([0, 0, 1, 0], index=data.index, name="state", dtype="int64")
    changed = pd.Series([0, 1, 1, 0], index=data.index, name="state", dtype="int64")
    base_comparison, _ = build_dual_method_comparison(data, baseline, _statistics())
    changed_comparison, _ = build_dual_method_comparison(data, changed, _statistics())

    for column in ("hmm_100_keep", "hmm_60_40_spread"):
        assert changed_comparison.iloc[0][column] == base_comparison.iloc[0][column]
        assert changed_comparison.iloc[1][column] != base_comparison.iloc[1][column]
        assert changed_comparison.iloc[2][column] == base_comparison.iloc[2][column]
