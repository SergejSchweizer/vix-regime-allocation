"""Market-data loading for the VIX regime-allocation project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.api.types import is_numeric_dtype

TICKERS: dict[str, str] = {
    "TLT": "TLT",
    "GLD": "GLD",
    "SPY": "SPY",
    "VIX": "^VIX",
}

_INTERNAL_COLUMNS = tuple(TICKERS)
_YAHOO_TICKERS = tuple(TICKERS.values())


def _extract_adjusted_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract the adjusted-close cross section from a yfinance download."""
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Yahoo download must expose a MultiIndex containing 'Adj Close'.")

    for level in range(raw.columns.nlevels):
        level_values = raw.columns.get_level_values(level)
        if "Adj Close" in level_values:
            adjusted = raw.xs("Adj Close", axis=1, level=level, drop_level=True)
            if not isinstance(adjusted, pd.DataFrame):
                raise ValueError("Adjusted-close extraction must produce a DataFrame.")
            if adjusted.columns.has_duplicates:
                raise ValueError("Adjusted-close data contains duplicate ticker columns.")
            return adjusted.copy()

    raise ValueError("Yahoo download is missing the required 'Adj Close' field.")


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the exact canonical Date index contract."""
    normalized = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    except (TypeError, ValueError) as exc:
        raise ValueError("Yahoo download index must be convertible to DatetimeIndex.") from exc

    if index.hasnans:
        raise ValueError("Yahoo download index contains an invalid/NaT date.")
    if index.tz is not None:
        index = index.tz_localize(None)

    normalized.index = index
    normalized.index.name = "Date"
    if normalized.index.has_duplicates:
        raise ValueError("Yahoo download contains duplicate dates.")

    return normalized.sort_index()


def _validate_prices(frame: pd.DataFrame) -> None:
    """Validate non-missing adjusted-close values without removing missing rows."""
    for column in frame.columns:
        if not is_numeric_dtype(frame[column].dtype):
            raise ValueError(f"Adjusted-close column {column!r} must be numeric.")

    values = frame.to_numpy(dtype=float)
    present = ~np.isnan(values)
    if np.any(~np.isfinite(values[present])):
        raise ValueError("Non-missing adjusted-close values must be finite.")
    if np.any(values[present] <= 0.0):
        raise ValueError("Non-missing adjusted-close values must be strictly positive.")


def download_adjusted_close() -> pd.DataFrame:
    """Download canonical adjusted-close histories for TLT, GLD, SPY and VIX.

    Missing prices are intentionally preserved. PR-02 is responsible for taking
    the common-date intersection before computing returns and VIX changes.
    """
    raw = yf.download(
        tickers=list(_YAHOO_TICKERS),
        period="max",
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=False,
        progress=False,
    )
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("yfinance.download must return a pandas DataFrame.")

    adjusted = _extract_adjusted_close(raw)
    if adjusted.empty:
        raise ValueError("Yahoo download returned no adjusted-close observations.")

    missing_tickers = [ticker for ticker in _YAHOO_TICKERS if ticker not in adjusted.columns]
    if missing_tickers:
        raise ValueError(f"Adjusted-close data missing required tickers: {missing_tickers}.")

    adjusted = adjusted.loc[:, list(_YAHOO_TICKERS)].rename(
        columns={external: internal for internal, external in TICKERS.items()}
    )
    adjusted = adjusted.loc[:, list(_INTERNAL_COLUMNS)]
    adjusted = _normalize_index(adjusted)
    _validate_prices(adjusted)
    return adjusted
