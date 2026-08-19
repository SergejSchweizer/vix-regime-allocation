"""Step 5 preferred-family K=2 versus K=3 state-count sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .allocation import build_state_allocation
from .backtest import build_rotation_returns
from .performance import PERFORMANCE_KEYS, performance_metrics
from .state_statistics import compute_state_asset_statistics
from .transform import OUTPUT_COLUMNS

SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "family",
    "n_states",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)
SUPPORTED_FAMILIES: tuple[str, str] = ("markov", "hmm")


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
    if len(data) < 4:
        raise ValueError(
            "data must contain enough observations for two- and three-state sensitivity."
        )
    for column in OUTPUT_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"data column {column!r} must be numeric.")
    if np.any(~np.isfinite(data.to_numpy(dtype=float))):
        raise ValueError("data must contain only finite values.")


def _validate_states(states: pd.Series, data_index: pd.DatetimeIndex, n_states: int) -> pd.Series:
    if not isinstance(states, pd.Series):
        raise TypeError(f"states_by_k[{n_states}] must be a pandas Series.")
    if states.name != "state":
        raise ValueError(f"states_by_k[{n_states}] must be named 'state'.")
    if not states.index.equals(data_index):
        raise ValueError(f"states_by_k[{n_states}] must use exactly the Step 1 Date index.")
    if not is_integer_dtype(states.dtype):
        raise ValueError(f"states_by_k[{n_states}] must use an integer dtype.")
    values = states.to_numpy(dtype=int)
    unique = np.unique(values)
    if not np.array_equal(unique, np.arange(n_states, dtype=int)):
        raise ValueError(
            f"states_by_k[{n_states}] must contain contiguous labels 0..{n_states - 1}."
        )
    counts = np.bincount(values, minlength=n_states)
    if np.any(counts < 2):
        raise ValueError(f"states_by_k[{n_states}] requires at least two observations per state.")
    return states


def build_state_count_sensitivity(
    data: pd.DataFrame, preferred_family: str, states_by_k: dict[int, pd.Series]
) -> pd.DataFrame:
    """Compare K=2 and K=3 within one already-selected model family.

    The function never fits or decodes a regime model. It reuses canonical state
    sequences and delegates state statistics, allocation, lagged rotation, and
    performance metrics to the shared Step 3–5 functions. Metrics are computed only
    on the common lagged return-date intersection across K=2 and K=3.
    """
    _validate_data(data)
    if preferred_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"preferred_family must be one of {SUPPORTED_FAMILIES}.")
    if not isinstance(states_by_k, dict):
        raise TypeError("states_by_k must be a dict keyed by 2 and 3.")
    if set(states_by_k) != {2, 3}:
        raise ValueError("states_by_k must contain exactly keys 2 and 3.")

    data_index = pd.DatetimeIndex(data.index, name="Date")
    rotations: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        states = _validate_states(states_by_k[n_states], data_index, n_states)
        statistics = compute_state_asset_statistics(data, states)
        allocation = build_state_allocation(statistics)
        rotation_detail = build_rotation_returns(data, states, allocation)
        rotations[n_states] = rotation_detail["regime_rotation_return"].rename(
            f"regime_rotation_k{n_states}"
        )

    common_index = rotations[2].index.intersection(rotations[3].index, sort=False)
    if len(common_index) < 2:
        raise ValueError("K=2 and K=3 rotations must share at least two return dates.")
    if not common_index.is_monotonic_increasing:
        common_index = common_index.sort_values()

    rows: list[dict[str, float | int | str]] = []
    for n_states in (2, 3):
        common_returns = rotations[n_states].reindex(common_index)
        if common_returns.isna().any():
            raise ValueError(
                "common sensitivity return dates must be complete for both state counts."
            )
        metrics = performance_metrics(common_returns)
        if tuple(metrics.keys()) != PERFORMANCE_KEYS:
            raise ValueError("performance_metrics returned an unexpected metric schema.")
        rows.append({"family": preferred_family, "n_states": n_states, **metrics})

    return pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))
