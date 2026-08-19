from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.transform import OUTPUT_COLUMNS, prepare_step1_data


def _prices() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], name="Date")
    return pd.DataFrame(
        {
            "TLT": [100.0, 110.0, np.nan, 121.0],
            "GLD": [50.0, 55.0, 60.0, 60.5],
            "SPY": [200.0, 220.0, 240.0, 242.0],
            "VIX": [10.0, 12.0, 15.0, 11.0],
        },
        index=index,
    )


def test_prepare_step1_data_drops_incomplete_dates_before_lags() -> None:
    result = prepare_step1_data(_prices())

    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert list(result.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-04")]
    assert result.index.name == "Date"

    expected_first = np.log(1.1)
    expected_second = np.log(1.1)
    assert result.loc["2020-01-02", "TLT_log_return"] == pytest.approx(expected_first)
    assert result.loc["2020-01-04", "TLT_log_return"] == pytest.approx(expected_second)
    assert result.loc["2020-01-02", "GLD_log_return"] == pytest.approx(np.log(55.0 / 50.0))
    assert result.loc["2020-01-04", "GLD_log_return"] == pytest.approx(np.log(60.5 / 55.0))
    assert result.loc["2020-01-02", "SPY_log_return"] == pytest.approx(np.log(220.0 / 200.0))
    assert result.loc["2020-01-04", "SPY_log_return"] == pytest.approx(np.log(242.0 / 220.0))
    assert result.loc["2020-01-02", "VIX_change"] == pytest.approx(2.0)
    assert result.loc["2020-01-04", "VIX_change"] == pytest.approx(-1.0)
    assert np.isfinite(result.to_numpy(dtype=float)).all()


def test_prepare_step1_data_requires_exact_columns() -> None:
    prices = _prices().rename(columns={"VIX": "^VIX"})
    with pytest.raises(ValueError, match="columns must be exactly"):
        prepare_step1_data(prices)


def test_prepare_step1_data_requires_canonical_datetime_index() -> None:
    unnamed = _prices().rename_axis(None)
    with pytest.raises(ValueError, match="named 'Date'"):
        prepare_step1_data(unnamed)

    duplicated = _prices().copy()
    duplicated.index = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-04"], name="Date"
    )
    with pytest.raises(ValueError, match="duplicate"):
        prepare_step1_data(duplicated)

    unsorted = _prices().iloc[[1, 0, 2, 3]]
    with pytest.raises(ValueError, match="sorted"):
        prepare_step1_data(unsorted)


def test_prepare_step1_data_rejects_non_numeric_or_invalid_prices() -> None:
    non_numeric = _prices().copy()
    non_numeric["SPY"] = non_numeric["SPY"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        prepare_step1_data(non_numeric)

    non_finite = _prices().copy()
    non_finite.loc["2020-01-02", "GLD"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        prepare_step1_data(non_finite)

    non_positive = _prices().copy()
    non_positive.loc["2020-01-02", "TLT"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_step1_data(non_positive)


def test_prepare_step1_data_requires_two_complete_observations() -> None:
    prices = _prices().copy()
    prices.loc[["2020-01-02", "2020-01-04"], "VIX"] = np.nan
    with pytest.raises(ValueError, match="At least two complete"):
        prepare_step1_data(prices)


def test_prepare_step1_data_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        prepare_step1_data([])  # type: ignore[arg-type]
