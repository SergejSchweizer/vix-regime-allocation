"""State-conditional ETF return statistics for Step 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .model_config import SUPPORTED_STATE_COUNTS
from .transform import OUTPUT_COLUMNS

ASSET_ORDER: tuple[str, str, str] = ("TLT", "GLD", "SPY")
STATISTICS_COLUMNS: tuple[str, ...] = (
    "state",
    "asset",
    "mean_log_return",
    "std_log_return",
    "observations",
)


def _validate_data(data: pd.DataFrame) -> None:
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
        raise ValueError("data must contain at least one observation.")
    for column in OUTPUT_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"data column {column!r} must be numeric.")
    if np.any(~np.isfinite(data.to_numpy(dtype=float))):
        raise ValueError("data must contain only finite values.")


def _validate_states(states: pd.Series, index: pd.Index) -> int:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series.")
    if states.name != "state":
        raise ValueError("states Series must be named 'state'.")
    if not states.index.equals(index):
        raise ValueError("states must use exactly the same Date index as data.")
    if not is_integer_dtype(states.dtype):
        raise ValueError("states must use an integer dtype.")
    values = states.to_numpy(dtype=int)
    if len(values) == 0:
        raise ValueError("states must contain observations.")
    unique = np.unique(values)
    if len(unique) not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"states must contain exactly one of {SUPPORTED_STATE_COUNTS} regimes.")
    n_states = int(len(unique))
    if not np.array_equal(unique, np.arange(n_states, dtype=int)):
        raise ValueError("states must be contiguous integer labels starting at zero.")
    counts = np.bincount(values, minlength=n_states)
    if np.any(counts < 2):
        raise ValueError(
            "Each state requires at least two observations for sample standard deviation."
        )
    return n_states


def compute_state_asset_statistics(data: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """Compute daily mean log return, sample std (ddof=1), and count by state and ETF."""
    _validate_data(data)
    n_states = _validate_states(states, data.index)

    rows: list[dict[str, object]] = []
    for state in range(n_states):
        mask = states == state
        for asset in ASSET_ORDER:
            returns = data.loc[mask, f"{asset}_log_return"].astype(float)
            rows.append(
                {
                    "state": state,
                    "asset": asset,
                    "mean_log_return": float(returns.mean()),
                    "std_log_return": float(returns.std(ddof=1)),
                    "observations": int(returns.count()),
                }
            )

    result = pd.DataFrame(rows, columns=list(STATISTICS_COLUMNS))
    numeric = result[["mean_log_return", "std_log_return"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Computed state statistics must be finite.")
    return result
