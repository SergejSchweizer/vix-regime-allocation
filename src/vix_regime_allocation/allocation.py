"""Deterministic Step 4 mapping from regime statistics to 100% ETF allocations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .state_statistics import ASSET_ORDER, STATISTICS_COLUMNS

ALLOCATION_COLUMNS: tuple[str, ...] = (
    "state",
    "selected_asset",
    "selection_mean_log_return",
    "TLT_weight",
    "GLD_weight",
    "SPY_weight",
)


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


def build_state_allocation(statistics: pd.DataFrame) -> pd.DataFrame:
    """Choose the maximum conditional-mean ETF per state with fixed tie priority.

    The assignment implementation uses a 100% allocation to the selected ETF. Exact
    equal means are resolved deterministically by the fixed priority TLT -> GLD -> SPY.
    """
    n_states = _validate_statistics(statistics)
    rows: list[dict[str, object]] = []

    for state in range(n_states):
        state_rows = statistics.loc[statistics["state"] == state].set_index("asset")
        means = {asset: float(state_rows.loc[asset, "mean_log_return"]) for asset in ASSET_ORDER}
        best_mean = max(means.values())
        selected_asset = next(asset for asset in ASSET_ORDER if means[asset] == best_mean)
        rows.append(
            {
                "state": state,
                "selected_asset": selected_asset,
                "selection_mean_log_return": best_mean,
                "TLT_weight": 1.0 if selected_asset == "TLT" else 0.0,
                "GLD_weight": 1.0 if selected_asset == "GLD" else 0.0,
                "SPY_weight": 1.0 if selected_asset == "SPY" else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=list(ALLOCATION_COLUMNS))
