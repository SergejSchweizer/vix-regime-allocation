"""Predictive simple-return and individual-asset benchmark primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import ASSET_ORDER

_LOG_COLUMNS = tuple(f"{asset}_log_return" for asset in ASSET_ORDER)


def asset_simple_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the canonical ETF log returns to simple returns."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    missing = [column for column in _LOG_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing ETF log-return columns: {missing}.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a DatetimeIndex.")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("data dates must be unique and sorted ascending.")
    for column in _LOG_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"{column} must be numeric.")
    values = np.expm1(data.loc[:, list(_LOG_COLUMNS)].to_numpy(dtype=float))
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("simple ETF returns must be finite and strictly greater than -1.")
    result = pd.DataFrame(values, index=data.index.copy(), columns=list(ASSET_ORDER))
    result.index.name = data.index.name
    return result


def buy_and_hold_returns(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Return the identical-date TLT/GLD/SPY buy-and-hold return matrix."""

    if not isinstance(simple_returns, pd.DataFrame):
        raise TypeError("simple_returns must be a pandas DataFrame.")
    if tuple(simple_returns.columns) != ASSET_ORDER:
        raise ValueError("simple_returns columns must be exactly TLT, GLD, SPY.")
    values = simple_returns.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("buy-and-hold returns must be finite and greater than -1.")
    return simple_returns.copy()
