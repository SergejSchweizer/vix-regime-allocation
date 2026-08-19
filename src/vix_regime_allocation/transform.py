"""Deterministic Step 1 common-sample transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PRICE_COLUMNS: tuple[str, ...] = ("TLT", "GLD", "SPY", "VIX")
ETF_COLUMNS: tuple[str, ...] = ("TLT", "GLD", "SPY")
OUTPUT_COLUMNS: tuple[str, ...] = (
    "TLT",
    "GLD",
    "SPY",
    "VIX",
    "TLT_log_return",
    "GLD_log_return",
    "SPY_log_return",
    "VIX_change",
)


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate the exact loader-to-transformer interface contract."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if tuple(prices.columns) != PRICE_COLUMNS:
        raise ValueError(f"prices columns must be exactly {list(PRICE_COLUMNS)}.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a pandas DatetimeIndex.")
    if prices.index.name != "Date":
        raise ValueError("prices index must be named 'Date'.")
    if prices.index.tz is not None:
        raise ValueError("prices index must be timezone-naive.")
    if prices.index.has_duplicates:
        raise ValueError("prices index must not contain duplicate dates.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted in ascending order.")

    for column in PRICE_COLUMNS:
        if not is_numeric_dtype(prices[column].dtype):
            raise ValueError(f"prices column {column!r} must be numeric.")

    values = prices.to_numpy(dtype=float)
    present = ~np.isnan(values)
    if np.any(~np.isfinite(values[present])):
        raise ValueError("Non-missing prices must be finite.")
    if np.any(values[present] <= 0.0):
        raise ValueError("Non-missing prices must be strictly positive.")


def prepare_step1_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Create the exact common-sample Step 1 dataset.

    Dates missing any required price are removed before lagged quantities are
    computed. No interpolation or filling is performed. ETF returns are daily
    log returns over consecutive rows of the resulting common sample, while the
    VIX observation is the first difference of the VIX level.
    """
    _validate_prices(prices)

    common = prices.dropna(subset=list(PRICE_COLUMNS)).copy()
    if len(common) < 2:
        raise ValueError("At least two complete common-date observations are required.")

    result = common.copy()
    for asset in ETF_COLUMNS:
        result[f"{asset}_log_return"] = np.log(result[asset] / result[asset].shift(1))
    result["VIX_change"] = result["VIX"].diff()

    result = result.iloc[1:].loc[:, list(OUTPUT_COLUMNS)].copy()
    values = result.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Prepared Step 1 data contains missing or non-finite values.")

    return result
