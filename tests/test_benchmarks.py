from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.benchmarks import (
    EQUAL_WEIGHT_NAME,
    SPY_NAME,
    build_equal_weight_monthly_returns,
    build_spy_buy_hold_returns,
)
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _data() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"], name="Date")
    frame = pd.DataFrame(index=index)
    frame["TLT"] = [100.0, 101.0, 102.0, 103.0]
    frame["GLD"] = [200.0, 201.0, 202.0, 203.0]
    frame["SPY"] = [300.0, 301.0, 302.0, 303.0]
    frame["VIX"] = [20.0, 21.0, 19.0, 18.0]
    simple = {
        "TLT": [0.10, 0.00, 0.03, 0.02],
        "GLD": [0.00, 0.00, 0.00, -0.01],
        "SPY": [0.00, 0.10, 0.00, 0.04],
    }
    for asset in ("TLT", "GLD", "SPY"):
        frame[f"{asset}_log_return"] = np.log1p(simple[asset])
    frame["VIX_change"] = [0.0, 1.0, -2.0, -1.0]
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def test_equal_weight_resets_monthly_and_drifts_intra_month() -> None:
    data = _data()
    index = pd.DatetimeIndex(data.index, name="Date")

    result = build_equal_weight_monthly_returns(data, index)

    expected_first = 0.10 / 3.0
    w_after_first = np.array([1.1, 1.0, 1.0]) / 3.1
    expected_second = float(np.dot(w_after_first, np.array([0.0, 0.0, 0.1])))
    expected_third = 0.03 / 3.0
    w_after_third = np.array([1.03, 1.0, 1.0]) / 3.03
    expected_fourth = float(np.dot(w_after_third, np.array([0.02, -0.01, 0.04])))

    assert result.name == EQUAL_WEIGHT_NAME
    assert result.index.equals(index)
    np.testing.assert_allclose(
        result.to_numpy(), [expected_first, expected_second, expected_third, expected_fourth]
    )
    assert result.iloc[1] != pytest.approx((0.0 + 0.0 + 0.1) / 3.0)


def test_spy_buy_hold_is_exact_spy_simple_return() -> None:
    data = _data()
    index = pd.DatetimeIndex(data.index, name="Date")
    result = build_spy_buy_hold_returns(data, index)

    assert result.name == SPY_NAME
    assert result.index.equals(index)
    np.testing.assert_allclose(result.to_numpy(), [0.0, 0.1, 0.0, 0.04])


def test_benchmarks_accept_exact_sorted_subset_comparison_index() -> None:
    data = _data()
    subset = pd.DatetimeIndex(data.index[1:], name="Date")

    equal_weight = build_equal_weight_monthly_returns(data, subset)
    spy = build_spy_buy_hold_returns(data, subset)

    assert equal_weight.index.equals(subset)
    assert spy.index.equals(subset)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "unsorted", "wrong_name", "timezone", "empty"]
)
def test_benchmarks_reject_invalid_comparison_index(case: str) -> None:
    data = _data()
    if case == "missing":
        index = pd.DatetimeIndex(["2026-01-29", "2026-03-01"], name="Date")
    elif case == "duplicate":
        index = pd.DatetimeIndex(["2026-01-29", "2026-01-29"], name="Date")
    elif case == "unsorted":
        index = pd.DatetimeIndex(["2026-01-30", "2026-01-29"], name="Date")
    elif case == "wrong_name":
        index = pd.DatetimeIndex(data.index, name="date")
    elif case == "timezone":
        index = pd.DatetimeIndex(data.index, name="Date").tz_localize("UTC")
    else:
        index = pd.DatetimeIndex([], name="Date")

    with pytest.raises(ValueError):
        build_equal_weight_monthly_returns(data, index)
    with pytest.raises(ValueError):
        build_spy_buy_hold_returns(data, index)


def test_benchmarks_reject_non_datetime_comparison_index() -> None:
    data = _data()
    with pytest.raises(TypeError):
        build_equal_weight_monthly_returns(data, pd.Index([1, 2]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case", ["not_dataframe", "schema", "index_type", "index_name", "duplicate", "empty", "nonnumeric", "nonfinite"]
)
def test_benchmarks_reject_noncanonical_data(case: str) -> None:
    data: object = _data().copy()
    if case == "not_dataframe":
        data = []
    elif case == "schema":
        data = _data().drop(columns=["VIX_change"])
    elif case == "index_type":
        frame = _data().copy()
        frame.index = pd.Index(range(len(frame)), name="Date")
        data = frame
    elif case == "index_name":
        frame = _data().copy()
        frame.index = frame.index.rename("date")
        data = frame
    elif case == "duplicate":
        frame = _data().copy()
        frame.index = pd.DatetimeIndex(
            ["2026-01-29", "2026-01-29", "2026-02-02", "2026-02-03"], name="Date"
        )
        data = frame
    elif case == "empty":
        data = _data().iloc[0:0]
    elif case == "nonnumeric":
        frame = _data().copy()
        frame["TLT"] = "bad"
        data = frame
    elif case == "nonfinite":
        frame = _data().copy()
        frame.loc[frame.index[0], "TLT"] = np.inf
        data = frame

    index = pd.DatetimeIndex(_data().index, name="Date")
    error = TypeError if case == "not_dataframe" else ValueError
    with pytest.raises(error):
        build_equal_weight_monthly_returns(data, index)  # type: ignore[arg-type]


def test_benchmark_simple_return_validation_rejects_invalid_log_return() -> None:
    data = _data().copy()
    data.loc[data.index[0], "SPY_log_return"] = -np.inf
    index = pd.DatetimeIndex(data.index, name="Date")
    with pytest.raises(ValueError):
        build_spy_buy_hold_returns(data, index)
