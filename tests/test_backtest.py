from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.allocation import ALLOCATION_COLUMNS
from vix_regime_allocation.backtest import ROTATION_DETAIL_COLUMNS, build_rotation_returns
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _data() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"], name="Date"
    )
    simple = {
        "TLT": np.array([0.00, 0.10, 0.20, 0.30]),
        "GLD": np.array([0.00, 0.01, 0.02, 0.03]),
        "SPY": np.array([0.00, -0.05, 0.04, -0.02]),
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


def _allocation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": 0,
                "selected_asset": "SPY",
                "selection_mean_log_return": 0.01,
                "TLT_weight": 0.0,
                "GLD_weight": 0.0,
                "SPY_weight": 1.0,
            },
            {
                "state": 1,
                "selected_asset": "TLT",
                "selection_mean_log_return": 0.02,
                "TLT_weight": 1.0,
                "GLD_weight": 0.0,
                "SPY_weight": 0.0,
            },
        ],
        columns=list(ALLOCATION_COLUMNS),
    )


def test_build_rotation_returns_applies_exact_previous_observed_row_state() -> None:
    data = _data()
    states = pd.Series([0, 1, 1, 0], index=data.index, name="state", dtype=int)

    result = build_rotation_returns(data, states, _allocation())

    assert tuple(result.columns) == ROTATION_DETAIL_COLUMNS
    assert result.index.equals(data.index[1:])
    assert result.index.name == "Date"
    assert result["decision_date"].tolist() == data.index[:-1].tolist()
    assert result["decision_state"].tolist() == [0, 1, 1]
    assert result["selected_asset"].tolist() == ["SPY", "TLT", "TLT"]
    assert result[["TLT_weight", "GLD_weight", "SPY_weight"]].values.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    np.testing.assert_allclose(result["regime_rotation_return"], [-0.05, 0.20, 0.30])


def test_state_change_affects_only_next_trading_row() -> None:
    data = _data()
    baseline = pd.Series([0, 0, 0, 0], index=data.index, name="state", dtype=int)
    changed = pd.Series([0, 1, 0, 0], index=data.index, name="state", dtype=int)

    baseline_result = build_rotation_returns(data, baseline, _allocation())
    changed_result = build_rotation_returns(data, changed, _allocation())

    assert changed_result.loc[data.index[1], "regime_rotation_return"] == pytest.approx(
        baseline_result.loc[data.index[1], "regime_rotation_return"]
    )
    assert changed_result.loc[data.index[2], "regime_rotation_return"] == pytest.approx(0.20)
    assert changed_result.loc[data.index[3], "regime_rotation_return"] == pytest.approx(
        baseline_result.loc[data.index[3], "regime_rotation_return"]
    )


@pytest.mark.parametrize(
    "case",
    ["state_index", "state_name", "allocation_state", "allocation_weights", "data_schema"],
)
def test_build_rotation_returns_rejects_contract_violations(case: str) -> None:
    data = _data()
    states = pd.Series([0, 1, 1, 0], index=data.index, name="state", dtype=int)
    allocation = _allocation()

    if case == "state_index":
        states = states.iloc[::-1]
    elif case == "state_name":
        states = states.rename("regime")
    elif case == "allocation_state":
        allocation.loc[1, "state"] = 2
    elif case == "allocation_weights":
        allocation.loc[0, "TLT_weight"] = 1.0
    else:
        data = data.drop(columns=["VIX_change"])

    with pytest.raises(ValueError):
        build_rotation_returns(data, states, allocation)


def test_build_rotation_returns_rejects_non_dataframe_inputs() -> None:
    data = _data()
    states = pd.Series([0, 1, 1, 0], index=data.index, name="state", dtype=int)
    allocation = _allocation()

    with pytest.raises(TypeError):
        build_rotation_returns([], states, allocation)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_rotation_returns(data, [0, 1, 1, 0], allocation)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_rotation_returns(data, states, [])  # type: ignore[arg-type]
