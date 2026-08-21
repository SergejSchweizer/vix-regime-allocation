"""Step 5 HMM state-count and allocation-method sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .allocation import build_state_allocation
from .backtest import build_rotation_returns
from .performance import PERFORMANCE_KEYS, performance_metrics
from .state_statistics import compute_state_asset_statistics
from .transform import OUTPUT_COLUMNS

HMM_SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "family",
    "n_states",
    "method",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)
METHOD_ORDER: tuple[str, str] = ("100_keep", "60_40_spread")

# Transitional legacy contract retained until the full canonical rebuild migrates old callers.
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
        raise ValueError("data must contain enough observations for K=2/K=3 sensitivity.")
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
    if not np.array_equal(np.unique(values), np.arange(n_states, dtype=int)):
        raise ValueError(
            f"states_by_k[{n_states}] must contain contiguous labels 0..{n_states - 1}."
        )
    counts = np.bincount(values, minlength=n_states)
    if np.any(counts < 2):
        raise ValueError(f"states_by_k[{n_states}] requires at least two observations per state.")
    return states


def _validate_state_dictionary(
    data: pd.DataFrame, states_by_k: dict[int, pd.Series]
) -> dict[int, pd.Series]:
    _validate_data(data)
    if not isinstance(states_by_k, dict):
        raise TypeError("states_by_k must be a dict keyed by 2 and 3.")
    if set(states_by_k) != {2, 3}:
        raise ValueError("states_by_k must contain exactly keys 2 and 3.")
    index = pd.DatetimeIndex(data.index, name="Date")
    return {k: _validate_states(states_by_k[k], index, k) for k in (2, 3)}


def build_hmm_state_count_sensitivity(
    data: pd.DataFrame, states_by_k: dict[int, pd.Series]
) -> pd.DataFrame:
    """Return exactly HMM K=2/K=3 crossed with 100% Keep and 60/40 Spread.

    The function does not fit or decode a model. It reuses supplied canonical HMM state
    paths and delegates state statistics, ranking/allocation, lagged portfolio returns,
    and performance metrics to the shared implementation.
    """
    validated = _validate_state_dictionary(data, states_by_k)
    rotations: dict[tuple[int, str], pd.Series] = {}
    for n_states in (2, 3):
        statistics = compute_state_asset_statistics(data, validated[n_states])
        for method in METHOD_ORDER:
            allocation = build_state_allocation(statistics, method)
            detail = build_rotation_returns(data, validated[n_states], allocation)
            rotations[(n_states, method)] = detail["regime_rotation_return"].rename(
                f"hmm_k{n_states}_{method}"
            )

    common_index: pd.DatetimeIndex | None = None
    for returns in rotations.values():
        returns_index = pd.DatetimeIndex(returns.index, name="Date")
        if common_index is None:
            common_index = returns_index
        else:
            common_index = common_index.intersection(returns_index, sort=False)
    if common_index is None or len(common_index) < 2:
        raise ValueError("all four HMM sensitivity paths must share at least two return dates.")
    if not common_index.is_monotonic_increasing:
        common_index = common_index.sort_values()

    rows: list[dict[str, float | int | str]] = []
    for n_states in (2, 3):
        for method in METHOD_ORDER:
            returns = rotations[(n_states, method)].reindex(common_index)
            if returns.isna().any():
                raise ValueError("common sensitivity dates must be complete for all four paths.")
            metrics = performance_metrics(returns)
            if tuple(metrics.keys()) != PERFORMANCE_KEYS:
                raise ValueError("performance_metrics returned an unexpected metric schema.")
            rows.append(
                {
                    "family": "hmm",
                    "n_states": n_states,
                    "method": method,
                    **metrics,
                }
            )
    return pd.DataFrame(rows, columns=list(HMM_SENSITIVITY_COLUMNS))


def build_state_count_sensitivity(
    data: pd.DataFrame, preferred_family: str, states_by_k: dict[int, pd.Series]
) -> pd.DataFrame:
    """Temporary pre-revision two-row family adapter retained for old callers."""
    if preferred_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"preferred_family must be one of {SUPPORTED_FAMILIES}.")
    validated = _validate_state_dictionary(data, states_by_k)
    rotations: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        statistics = compute_state_asset_statistics(data, validated[n_states])
        allocation = build_state_allocation(statistics)
        detail = build_rotation_returns(data, validated[n_states], allocation)
        rotations[n_states] = detail["regime_rotation_return"].rename(
            f"regime_rotation_k{n_states}"
        )
    common_index = rotations[2].index.intersection(rotations[3].index, sort=False)
    if len(common_index) < 2:
        raise ValueError("K=2 and K=3 rotations must share at least two return dates.")
    rows: list[dict[str, float | int | str]] = []
    for n_states in (2, 3):
        metrics = performance_metrics(rotations[n_states].reindex(common_index))
        if tuple(metrics.keys()) != PERFORMANCE_KEYS:
            raise ValueError("performance_metrics returned an unexpected metric schema.")
        rows.append({"family": preferred_family, "n_states": n_states, **metrics})
    return pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))
