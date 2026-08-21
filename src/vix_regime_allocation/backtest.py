"""Step 5 regime-rotation returns with an exact one-observed-row execution lag."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from .allocation import ALLOCATION_COLUMNS, METHOD_ALLOCATION_COLUMNS
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


def _validate_state_order(frame: pd.DataFrame, n_states: int) -> pd.DataFrame:
    if len(frame) != n_states:
        raise ValueError("allocation must contain exactly one row for each state.")
    if not is_integer_dtype(frame["state"].dtype):
        raise ValueError("allocation state must use an integer dtype.")
    ordered = frame.sort_values("state").reset_index(drop=True)
    if not np.array_equal(ordered["state"].to_numpy(dtype=int), np.arange(n_states)):
        raise ValueError("allocation states must be contiguous labels 0..K-1.")
    return ordered


def _validate_weight_matrix(ordered: pd.DataFrame) -> np.ndarray:
    for column in ("TLT_weight", "GLD_weight", "SPY_weight"):
        if not is_numeric_dtype(ordered[column].dtype):
            raise ValueError(f"allocation column {column!r} must be numeric.")
    weights = ordered[["TLT_weight", "GLD_weight", "SPY_weight"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("allocation weights must be finite and non-negative.")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("allocation weights must sum to one within 1e-12.")
    return weights


def _validate_legacy_allocation(allocation: pd.DataFrame, n_states: int) -> pd.DataFrame:
    ordered = _validate_state_order(allocation, n_states)
    if not ordered["selected_asset"].isin(ASSET_ORDER).all():
        raise ValueError("selected_asset must be TLT, GLD, or SPY.")
    if not is_numeric_dtype(ordered["selection_mean_log_return"].dtype):
        raise ValueError("selection_mean_log_return must be numeric.")
    if not np.isfinite(ordered["selection_mean_log_return"].to_numpy(dtype=float)).all():
        raise ValueError("selection_mean_log_return must be finite.")
    weights = _validate_weight_matrix(ordered)
    if np.any((weights != 0.0) & (weights != 1.0)):
        raise ValueError("legacy allocation weights must remain one-hot.")
    for row in ordered.itertuples(index=False):
        selected = str(row.selected_asset)
        expected = np.array([1.0 if asset == selected else 0.0 for asset in ASSET_ORDER])
        actual = np.array([row.TLT_weight, row.GLD_weight, row.SPY_weight], dtype=float)
        if not np.array_equal(actual, expected):
            raise ValueError("selected_asset must agree with legacy one-hot allocation weights.")
    return ordered


def _validate_method_allocation(allocation: pd.DataFrame, n_states: int) -> pd.DataFrame:
    ordered = _validate_state_order(allocation, n_states)
    methods = ordered["method"].astype(str).unique().tolist()
    if len(methods) != 1 or methods[0] not in ("100_keep", "60_40_spread"):
        raise ValueError("method-aware allocation must contain one supported method only.")
    method = methods[0]
    for column in ("rank_1_asset", "rank_2_asset"):
        if not ordered[column].isin(ASSET_ORDER).all():
            raise ValueError(f"{column} must contain TLT, GLD, or SPY.")
    if (ordered["rank_1_asset"] == ordered["rank_2_asset"]).any():
        raise ValueError("rank_1_asset and rank_2_asset must differ in every state.")
    for column in ("rank_1_mean_log_return", "rank_2_mean_log_return"):
        if not is_numeric_dtype(ordered[column].dtype):
            raise ValueError(f"{column} must be numeric.")
        if not np.isfinite(ordered[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite.")
    if (ordered["rank_1_mean_log_return"] < ordered["rank_2_mean_log_return"]).any():
        raise ValueError("rank_1_mean_log_return must be at least rank_2_mean_log_return.")
    weights = _validate_weight_matrix(ordered)
    expected_top = 1.0 if method == "100_keep" else 0.6
    expected_second = 0.0 if method == "100_keep" else 0.4
    for position, row in enumerate(ordered.itertuples(index=False)):
        actual = {asset: float(weights[position, index]) for index, asset in enumerate(ASSET_ORDER)}
        if not np.isclose(actual[str(row.rank_1_asset)], expected_top, atol=1e-12, rtol=0.0):
            raise ValueError("rank-1 weight does not match the allocation method.")
        if not np.isclose(actual[str(row.rank_2_asset)], expected_second, atol=1e-12, rtol=0.0):
            raise ValueError("rank-2 weight does not match the allocation method.")
        remaining = set(ASSET_ORDER) - {str(row.rank_1_asset), str(row.rank_2_asset)}
        if any(not np.isclose(actual[asset], 0.0, atol=1e-12, rtol=0.0) for asset in remaining):
            raise ValueError("unranked asset weight must be zero.")
    normalized = pd.DataFrame(
        {
            "state": ordered["state"].astype(int),
            "selected_asset": ordered["rank_1_asset"].astype(str),
            "selection_mean_log_return": ordered["rank_1_mean_log_return"].astype(float),
            "TLT_weight": ordered["TLT_weight"].astype(float),
            "GLD_weight": ordered["GLD_weight"].astype(float),
            "SPY_weight": ordered["SPY_weight"].astype(float),
        },
        columns=list(ALLOCATION_COLUMNS),
    )
    return normalized


def _validate_allocation(allocation: pd.DataFrame, n_states: int) -> pd.DataFrame:
    if not isinstance(allocation, pd.DataFrame):
        raise TypeError("allocation must be a pandas DataFrame.")
    if tuple(allocation.columns) == ALLOCATION_COLUMNS:
        return _validate_legacy_allocation(allocation, n_states)
    if tuple(allocation.columns) == METHOD_ALLOCATION_COLUMNS:
        return _validate_method_allocation(allocation, n_states)
    raise ValueError("allocation columns must match a supported Step 4 schema exactly.")


def build_rotation_returns(
    data: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame
) -> pd.DataFrame:
    """Build daily regime-rotation returns using state ``t-1`` for return row ``t``.

    ETF log returns are converted to simple returns before portfolio arithmetic. The
    first data row is excluded because it has no previous observed trading-row state.
    Both 100% Keep and 60/40 Spread method-aware allocations are supported by the same
    lagged portfolio arithmetic.
    """
    _validate_data(data)
    date_index = pd.DatetimeIndex(data.index, name="Date")
    n_states = _validate_states(states, date_index)
    ordered_allocation = _validate_allocation(allocation, n_states)

    return_index = date_index[1:]
    decision_dates = date_index[:-1]
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
