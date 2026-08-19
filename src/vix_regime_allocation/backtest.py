"""Step 5 regime-rotation returns with an exact one-observed-row execution lag."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from .allocation import ALLOCATION_COLUMNS
from .state_statistics import ASSET_ORDER
from .transform import OUTPUT_COLUMNS

ROTATION_DETAIL_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "decision_state",
    "selected_asset",
    "TLT_weight",
    "GLD_weight",
    "SPY_weight",
    "regime_rotation_return",
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
    if len(data) < 2:
        raise ValueError("data must contain at least two observations.")
    for column in OUTPUT_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"data column {column!r} must be numeric.")
    if np.any(~np.isfinite(data.to_numpy(dtype=float))):
        raise ValueError("data must contain only finite values.")


def _validate_states(states: pd.Series, index: pd.DatetimeIndex) -> int:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series.")
    if states.name != "state":
        raise ValueError("states Series must be named 'state'.")
    if not states.index.equals(index):
        raise ValueError("states must use exactly the same Date index as data.")
    if not is_integer_dtype(states.dtype):
        raise ValueError("states must use an integer dtype.")
    unique = np.unique(states.to_numpy(dtype=int))
    if len(unique) not in (2, 3) or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("states must contain contiguous labels 0..K-1 for K=2 or K=3.")
    return int(len(unique))


def _validate_allocation(allocation: pd.DataFrame, n_states: int) -> pd.DataFrame:
    if not isinstance(allocation, pd.DataFrame):
        raise TypeError("allocation must be a pandas DataFrame.")
    if tuple(allocation.columns) != ALLOCATION_COLUMNS:
        raise ValueError("allocation columns must match the canonical Step 4 schema exactly.")
    if len(allocation) != n_states:
        raise ValueError("allocation must contain exactly one row for each state.")
    if not is_integer_dtype(allocation["state"].dtype):
        raise ValueError("allocation state must use an integer dtype.")
    ordered = allocation.sort_values("state").reset_index(drop=True)
    if not np.array_equal(ordered["state"].to_numpy(dtype=int), np.arange(n_states)):
        raise ValueError("allocation states must be contiguous labels 0..K-1.")
    if not ordered["selected_asset"].isin(ASSET_ORDER).all():
        raise ValueError("selected_asset must be TLT, GLD, or SPY.")
    for column in ("selection_mean_log_return", "TLT_weight", "GLD_weight", "SPY_weight"):
        if not is_numeric_dtype(ordered[column].dtype):
            raise ValueError(f"allocation column {column!r} must be numeric.")
    numeric = ordered[
        ["selection_mean_log_return", "TLT_weight", "GLD_weight", "SPY_weight"]
    ].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("allocation numeric values must be finite.")
    weights = ordered[["TLT_weight", "GLD_weight", "SPY_weight"]].to_numpy(dtype=float)
    if np.any((weights != 0.0) & (weights != 1.0)) or not np.allclose(
        weights.sum(axis=1), 1.0, rtol=0.0, atol=0.0
    ):
        raise ValueError("allocation weights must be one-hot and sum exactly to one.")
    for row in ordered.itertuples(index=False):
        selected = str(row.selected_asset)
        expected = np.array([1.0 if asset == selected else 0.0 for asset in ASSET_ORDER])
        actual = np.array([row.TLT_weight, row.GLD_weight, row.SPY_weight], dtype=float)
        if not np.array_equal(actual, expected):
            raise ValueError("selected_asset must agree with the one-hot allocation weights.")
    return ordered


def build_rotation_returns(
    data: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame
) -> pd.DataFrame:
    """Build daily regime-rotation returns using state ``t-1`` for return row ``t``.

    ETF log returns are converted to simple returns before portfolio arithmetic. The
    first data row is excluded because it has no previous observed trading-row state.
    """
    _validate_data(data)
    n_states = _validate_states(states, data.index)
    ordered_allocation = _validate_allocation(allocation, n_states)

    return_index = data.index[1:]
    decision_dates = data.index[:-1]
    decision_states = states.iloc[:-1].to_numpy(dtype=int)

    simple_returns = np.expm1(
        data.loc[return_index, [f"{asset}_log_return" for asset in ASSET_ORDER]].to_numpy(
            dtype=float
        )
    )
    if np.any(~np.isfinite(simple_returns)) or np.any(simple_returns <= -1.0):
        raise ValueError("converted ETF simple returns must be finite and greater than -1.")

    lookup = ordered_allocation.set_index("state")
    allocation_rows = lookup.loc[decision_states]
    weights = allocation_rows[["TLT_weight", "GLD_weight", "SPY_weight"]].to_numpy(dtype=float)
    rotation = np.sum(weights * simple_returns, axis=1)
    if np.any(~np.isfinite(rotation)) or np.any(rotation <= -1.0):
        raise ValueError("regime rotation returns must be finite and greater than -1.")

    result = pd.DataFrame(
        {
            "decision_date": decision_dates,
            "decision_state": decision_states,
            "selected_asset": allocation_rows["selected_asset"].to_numpy(dtype=object),
            "TLT_weight": weights[:, 0],
            "GLD_weight": weights[:, 1],
            "SPY_weight": weights[:, 2],
            "regime_rotation_return": rotation,
        },
        index=return_index,
        columns=list(ROTATION_DETAIL_COLUMNS),
    )
    result.index.name = "Date"
    if not is_datetime64_any_dtype(result["decision_date"].dtype):
        raise ValueError("decision_date must be datetime typed.")
    return result
