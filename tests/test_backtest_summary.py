from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.backtest_summary as module
from vix_regime_allocation.backtest import ROTATION_DETAIL_COLUMNS
from vix_regime_allocation.backtest_summary import (
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    build_comparison,
    build_performance_summary,
)
from vix_regime_allocation.benchmarks import EQUAL_WEIGHT_NAME, SPY_NAME
from vix_regime_allocation.performance import PERFORMANCE_KEYS


def _index() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-02", periods=3, freq="B", name="Date")


def _rotation() -> pd.DataFrame:
    index = _index()
    return pd.DataFrame(
        {
            "decision_date": pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-05"]),
            "decision_state": [0, 1, 0],
            "selected_asset": ["SPY", "TLT", "SPY"],
            "TLT_weight": [0.0, 1.0, 0.0],
            "GLD_weight": [0.0, 0.0, 0.0],
            "SPY_weight": [1.0, 0.0, 1.0],
            "regime_rotation_return": [0.01, -0.02, 0.03],
        },
        index=index,
        columns=list(ROTATION_DETAIL_COLUMNS),
    )


def _benchmarks() -> tuple[pd.Series, pd.Series]:
    index = _index()
    equal_weight = pd.Series([0.005, 0.01, -0.005], index=index, name=EQUAL_WEIGHT_NAME)
    spy = pd.Series([-0.01, 0.02, 0.01], index=index, name=SPY_NAME)
    return equal_weight, spy


def test_build_comparison_exact_schema_dates_and_values() -> None:
    equal_weight, spy = _benchmarks()
    result = build_comparison(_rotation(), equal_weight, spy)

    assert tuple(result.columns) == COMPARISON_COLUMNS
    assert result.index.equals(_index())
    assert result.index.name == "Date"
    np.testing.assert_allclose(result["regime_rotation"], [0.01, -0.02, 0.03])
    np.testing.assert_allclose(result[EQUAL_WEIGHT_NAME], equal_weight)
    np.testing.assert_allclose(result[SPY_NAME], spy)


def test_build_comparison_rejects_any_date_mismatch() -> None:
    equal_weight, spy = _benchmarks()
    shifted = equal_weight.copy()
    shifted.index = pd.date_range("2026-01-05", periods=3, freq="B", name="Date")
    with pytest.raises(ValueError, match="identical Date indexes"):
        build_comparison(_rotation(), shifted, spy)


def test_summary_delegates_exactly_once_per_portfolio(monkeypatch: pytest.MonkeyPatch) -> None:
    equal_weight, spy = _benchmarks()
    comparison = build_comparison(_rotation(), equal_weight, spy)
    calls: list[str] = []

    def fake_metrics(series: pd.Series) -> dict[str, float | int]:
        calls.append(str(series.name))
        base = float(len(calls))
        return {
            "cumulative_return": base,
            "annualized_return": base + 0.1,
            "annualized_volatility": base + 0.2,
            "sharpe_ratio": base + 0.3,
            "max_drawdown": -base,
            "observations": len(series),
        }

    monkeypatch.setattr(module, "performance_metrics", fake_metrics)
    result = build_performance_summary(comparison)

    assert calls == list(COMPARISON_COLUMNS)
    assert tuple(result.columns) == SUMMARY_COLUMNS
    assert result["portfolio"].tolist() == list(COMPARISON_COLUMNS)
    assert result["cumulative_return"].tolist() == [1.0, 2.0, 3.0]
    assert result["observations"].tolist() == [3, 3, 3]


def test_summary_rejects_unexpected_metric_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    equal_weight, spy = _benchmarks()
    comparison = build_comparison(_rotation(), equal_weight, spy)

    def malformed(_: pd.Series) -> dict[str, float | int]:
        return {key: 1.0 for key in PERFORMANCE_KEYS if key != "observations"}

    monkeypatch.setattr(module, "performance_metrics", malformed)
    with pytest.raises(ValueError, match="unexpected metric schema"):
        build_performance_summary(comparison)


@pytest.mark.parametrize(
    "case",
    ["type", "columns", "index_type", "index_name", "timezone", "duplicate", "unsorted", "nonnumeric", "nonfinite", "invalid_return"],
)
def test_build_comparison_rejects_invalid_rotation(case: str) -> None:
    rotation: object = _rotation().copy()
    if case == "type":
        rotation = []
    elif case == "columns":
        rotation = _rotation().drop(columns=["decision_state"])
    elif case == "index_type":
        frame = _rotation().copy()
        frame.index = pd.Index(range(len(frame)), name="Date")
        rotation = frame
    elif case == "index_name":
        frame = _rotation().copy()
        frame.index = frame.index.rename("date")
        rotation = frame
    elif case == "timezone":
        frame = _rotation().copy()
        frame.index = pd.DatetimeIndex(frame.index, name="Date").tz_localize("UTC")
        rotation = frame
    elif case == "duplicate":
        frame = _rotation().copy()
        frame.index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"], name="Date")
        rotation = frame
    elif case == "unsorted":
        rotation = _rotation().iloc[::-1]
    elif case == "nonnumeric":
        frame = _rotation().copy()
        frame["regime_rotation_return"] = "bad"
        rotation = frame
    elif case == "nonfinite":
        frame = _rotation().copy()
        frame.loc[frame.index[0], "regime_rotation_return"] = np.inf
        rotation = frame
    else:
        frame = _rotation().copy()
        frame.loc[frame.index[0], "regime_rotation_return"] = -1.0
        rotation = frame

    equal_weight, spy = _benchmarks()
    error = TypeError if case == "type" else ValueError
    with pytest.raises(error):
        build_comparison(rotation, equal_weight, spy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case", ["type", "name", "index_type", "index_name", "duplicate", "unsorted", "nonnumeric", "nonfinite", "invalid_return"]
)
def test_build_comparison_rejects_invalid_benchmark(case: str) -> None:
    equal_weight, spy = _benchmarks()
    benchmark: object = equal_weight.copy()
    if case == "type":
        benchmark = []
    elif case == "name":
        benchmark = equal_weight.rename("wrong")
    elif case == "index_type":
        series = equal_weight.copy()
        series.index = pd.Index(range(len(series)), name="Date")
        benchmark = series
    elif case == "index_name":
        series = equal_weight.copy()
        series.index = series.index.rename("date")
        benchmark = series
    elif case == "duplicate":
        series = equal_weight.copy()
        series.index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"], name="Date")
        benchmark = series
    elif case == "unsorted":
        benchmark = equal_weight.iloc[::-1]
    elif case == "nonnumeric":
        benchmark = pd.Series(["a", "b", "c"], index=_index(), name=EQUAL_WEIGHT_NAME)
    elif case == "nonfinite":
        series = equal_weight.copy()
        series.iloc[0] = np.inf
        benchmark = series
    else:
        series = equal_weight.copy()
        series.iloc[0] = -1.0
        benchmark = series

    error = TypeError if case == "type" else ValueError
    with pytest.raises(error):
        build_comparison(_rotation(), benchmark, spy)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["type", "columns", "index_type", "index_name", "duplicate", "unsorted"])
def test_summary_rejects_invalid_comparison(case: str) -> None:
    equal_weight, spy = _benchmarks()
    comparison: object = build_comparison(_rotation(), equal_weight, spy)
    if case == "type":
        comparison = []
    elif case == "columns":
        comparison = comparison.drop(columns=[SPY_NAME])  # type: ignore[union-attr]
    elif case == "index_type":
        frame = comparison.copy()  # type: ignore[union-attr]
        frame.index = pd.Index(range(len(frame)), name="Date")
        comparison = frame
    elif case == "index_name":
        frame = comparison.copy()  # type: ignore[union-attr]
        frame.index = frame.index.rename("date")
        comparison = frame
    elif case == "duplicate":
        frame = comparison.copy()  # type: ignore[union-attr]
        frame.index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"], name="Date")
        comparison = frame
    else:
        comparison = comparison.iloc[::-1]  # type: ignore[union-attr]

    error = TypeError if case == "type" else ValueError
    with pytest.raises(error):
        build_performance_summary(comparison)  # type: ignore[arg-type]
