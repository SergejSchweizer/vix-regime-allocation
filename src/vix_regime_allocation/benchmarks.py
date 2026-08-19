"""Step 5 benchmark return engines on an explicit common comparison index."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .state_statistics import ASSET_ORDER
from .transform import OUTPUT_COLUMNS

EQUAL_WEIGHT_NAME = "equal_weight_monthly"
SPY_NAME = "spy_buy_hold"


def _validate_data(data: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("data columns must match the canonical Step 1 schema exactly.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a pandas DatetimeIndex.")
    if data.index.name != "Date" or data.index.tz is not None:
        raise ValueError("data index must be timezone-naive and named 'Date'.")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("data dates must be unique and sorted ascending.")
    if len(data) == 0:
        raise ValueError("data must contain observations.")
    for column in OUTPUT_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"data column {column!r} must be numeric.")
    if np.any(~np.isfinite(data.to_numpy(dtype=float))):
        raise ValueError("data must contain only finite values.")
    return pd.DatetimeIndex(data.index, name="Date")


def _validate_comparison_index(
    comparison_index: pd.DatetimeIndex, data_index: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    if not isinstance(comparison_index, pd.DatetimeIndex):
        raise TypeError("comparison_index must be a pandas DatetimeIndex.")
    if comparison_index.name != "Date" or comparison_index.tz is not None:
        raise ValueError("comparison_index must be timezone-naive and named 'Date'.")
    if len(comparison_index) == 0:
        raise ValueError("comparison_index must contain observations.")
    if comparison_index.has_duplicates or not comparison_index.is_monotonic_increasing:
        raise ValueError("comparison_index must be unique and sorted ascending.")
    if not comparison_index.isin(data_index).all():
        raise ValueError("comparison_index must be an exact subset of the Step 1 Date index.")
    return comparison_index


def _simple_returns(data: pd.DataFrame, comparison_index: pd.DatetimeIndex) -> pd.DataFrame:
    log_columns = [f"{asset}_log_return" for asset in ASSET_ORDER]
    values = np.expm1(data.loc[comparison_index, log_columns].to_numpy(dtype=float))
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("benchmark simple returns must be finite and greater than -1.")
    return pd.DataFrame(values, index=comparison_index, columns=list(ASSET_ORDER))


def build_equal_weight_monthly_returns(
    data: pd.DataFrame, comparison_index: pd.DatetimeIndex
) -> pd.Series:
    """Return the monthly-reset 1/3 TLT, 1/3 GLD, 1/3 SPY benchmark.

    Weights are reset immediately before the first comparison return and before the
    first observed comparison return in each new calendar month. Between resets they
    drift according to realized simple returns.
    """
    data_index = _validate_data(data)
    index = _validate_comparison_index(comparison_index, data_index)
    returns = _simple_returns(data, index)

    weights = np.full(3, 1.0 / 3.0, dtype=float)
    portfolio_returns: list[float] = []
    previous_period: pd.Period | None = None

    for position in range(len(returns)):
        date = index[position]
        row = returns.iloc[position]
        period = date.to_period("M")
        if previous_period is None or period != previous_period:
            weights = np.full(3, 1.0 / 3.0, dtype=float)
        asset_returns = row.to_numpy(dtype=float)
        portfolio_return = float(np.dot(weights, asset_returns))
        if not np.isfinite(portfolio_return) or portfolio_return <= -1.0:
            raise ValueError("equal-weight benchmark return must be finite and greater than -1.")
        portfolio_returns.append(portfolio_return)
        weights = weights * (1.0 + asset_returns) / (1.0 + portfolio_return)
        if np.any(~np.isfinite(weights)) or not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                "drifted equal-weight benchmark weights must remain finite and sum to 1."
            )
        previous_period = period

    return pd.Series(portfolio_returns, index=index, name=EQUAL_WEIGHT_NAME, dtype=float)


def build_spy_buy_hold_returns(data: pd.DataFrame, comparison_index: pd.DatetimeIndex) -> pd.Series:
    """Return SPY simple returns on the exact comparison index."""
    data_index = _validate_data(data)
    index = _validate_comparison_index(comparison_index, data_index)
    spy = _simple_returns(data, index)["SPY"].copy()
    spy.name = SPY_NAME
    return spy
