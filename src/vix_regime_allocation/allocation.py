"""Deterministic Step 4 ranking for 100% Keep and 60/40 Spread allocations."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .state_statistics import ASSET_ORDER, STATISTICS_COLUMNS

# Transitional legacy schema used by old one-hot callers until the Step 5 engine is upgraded.
ALLOCATION_COLUMNS: tuple[str, ...] = (
    "state",
    "selected_asset",
    "selection_mean_log_return",
    "TLT_weight",
    "GLD_weight",
    "SPY_weight",
)

METHOD_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "method",
    "state",
    "rank_1_asset",
    "rank_2_asset",
    "rank_1_mean_log_return",
    "rank_2_mean_log_return",
    "TLT_weight",
    "GLD_weight",
    "SPY_weight",
)
SUPPORTED_ALLOCATION_METHODS: tuple[str, str] = ("100_keep", "60_40_spread")
_LEGACY_DEFAULT: Final[object] = object()


def _validate_statistics(statistics: pd.DataFrame) -> int:
    if not isinstance(statistics, pd.DataFrame):
        raise TypeError("statistics must be a pandas DataFrame.")
    if tuple(statistics.columns) != STATISTICS_COLUMNS:
        raise ValueError("statistics columns must match the canonical Step 3 schema exactly.")
    if len(statistics) == 0:
        raise ValueError("statistics must contain at least one state.")
    if not is_integer_dtype(statistics["state"].dtype):
        raise ValueError("state must use an integer dtype.")
    if statistics.duplicated(subset=["state", "asset"]).any():
        raise ValueError("statistics must contain each state/asset pair exactly once.")
    if not statistics["asset"].isin(ASSET_ORDER).all():
        raise ValueError("statistics assets must be exactly TLT, GLD, and SPY.")
    for column in ("mean_log_return", "std_log_return"):
        if not is_numeric_dtype(statistics[column].dtype):
            raise ValueError(f"{column} must be numeric.")
    if not is_integer_dtype(statistics["observations"].dtype):
        raise ValueError("observations must use an integer dtype.")
    numeric = statistics[["mean_log_return", "std_log_return"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("statistics means and standard deviations must be finite.")
    if (statistics["std_log_return"].astype(float) < 0.0).any():
        raise ValueError("statistics standard deviations must be non-negative.")
    if (statistics["observations"].astype(int) < 2).any():
        raise ValueError("each state/asset row requires at least two observations.")

    states = np.sort(statistics["state"].unique().astype(int))
    if len(states) not in (2, 3) or not np.array_equal(states, np.arange(len(states))):
        raise ValueError("statistics must contain contiguous states 0..K-1 for K=2 or K=3.")
    for state in states:
        subset = statistics.loc[statistics["state"] == state, "asset"].tolist()
        if sorted(subset, key=ASSET_ORDER.index) != list(ASSET_ORDER):
            raise ValueError("each state must contain TLT, GLD, and SPY exactly once.")
    return len(states)


def _rank_state(statistics: pd.DataFrame, state: int) -> list[tuple[str, float]]:
    state_rows = statistics.loc[statistics["state"] == state].set_index("asset")
    means = {asset: float(state_rows.loc[asset, "mean_log_return"]) for asset in ASSET_ORDER}
    priority = {asset: rank for rank, asset in enumerate(ASSET_ORDER)}
    return sorted(means.items(), key=lambda item: (-item[1], priority[item[0]]))


def _weights(method: str, rank_1: str, rank_2: str) -> dict[str, float]:
    if method == "100_keep":
        target = {rank_1: 1.0, rank_2: 0.0}
    elif method == "60_40_spread":
        target = {rank_1: 0.6, rank_2: 0.4}
    else:
        raise ValueError(f"method must be one of {SUPPORTED_ALLOCATION_METHODS}.")
    weights = {asset: float(target.get(asset, 0.0)) for asset in ASSET_ORDER}
    values = np.asarray([weights[asset] for asset in ASSET_ORDER], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError("allocation weights must be finite and non-negative.")
    if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("allocation weights must sum to one within 1e-12.")
    return weights


def _method_allocation(statistics: pd.DataFrame, method: str) -> pd.DataFrame:
    n_states = _validate_statistics(statistics)
    if method not in SUPPORTED_ALLOCATION_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_ALLOCATION_METHODS}.")
    rows: list[dict[str, object]] = []
    for state in range(n_states):
        ranking = _rank_state(statistics, state)
        rank_1_asset, rank_1_mean = ranking[0]
        rank_2_asset, rank_2_mean = ranking[1]
        weights = _weights(method, rank_1_asset, rank_2_asset)
        rows.append(
            {
                "method": method,
                "state": state,
                "rank_1_asset": rank_1_asset,
                "rank_2_asset": rank_2_asset,
                "rank_1_mean_log_return": rank_1_mean,
                "rank_2_mean_log_return": rank_2_mean,
                "TLT_weight": weights["TLT"],
                "GLD_weight": weights["GLD"],
                "SPY_weight": weights["SPY"],
            }
        )
    result = pd.DataFrame(rows, columns=list(METHOD_ALLOCATION_COLUMNS))
    sums = result[["TLT_weight", "GLD_weight", "SPY_weight"]].sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(sums, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("generated allocation rows must sum to one.")
    return result


def _legacy_100_keep(method_allocation: pd.DataFrame) -> pd.DataFrame:
    """Translate explicit 100% Keep rows to the historical one-hot schema."""
    if not (method_allocation["method"] == "100_keep").all():
        raise ValueError("legacy allocation translation supports 100_keep only.")
    return pd.DataFrame(
        {
            "state": method_allocation["state"].astype(int),
            "selected_asset": method_allocation["rank_1_asset"].astype(str),
            "selection_mean_log_return": method_allocation["rank_1_mean_log_return"].astype(float),
            "TLT_weight": method_allocation["TLT_weight"].astype(float),
            "GLD_weight": method_allocation["GLD_weight"].astype(float),
            "SPY_weight": method_allocation["SPY_weight"].astype(float),
        },
        columns=list(ALLOCATION_COLUMNS),
    )


def build_state_allocation(
    statistics: pd.DataFrame, method: str | object = _LEGACY_DEFAULT
) -> pd.DataFrame:
    """Rank state-conditional ETF means and build one of the two mandatory allocations.

    Explicit ``method='100_keep'`` and ``method='60_40_spread'`` calls return the new
    method-aware canonical schema. A temporarily supported omitted-method call returns
    the historical 100% Keep schema so pre-revision callers remain green until their
    dedicated backlog PR migrates them.
    """
    if method is _LEGACY_DEFAULT:
        return _legacy_100_keep(_method_allocation(statistics, "100_keep"))
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    return _method_allocation(statistics, method)
