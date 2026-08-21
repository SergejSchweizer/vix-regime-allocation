"""Step 5 comparison for both mandatory HMM allocation methods and benchmarks."""

from __future__ import annotations

import pandas as pd

from .allocation import build_state_allocation
from .backtest import build_rotation_returns
from .benchmarks import build_equal_weight_monthly_returns, build_spy_buy_hold_returns

COMPARISON_COLUMNS: tuple[str, str, str, str] = (
    "hmm_100_keep",
    "hmm_60_40_spread",
    "equal_weight_monthly",
    "spy_buy_hold",
)
ROTATION_METHODS: tuple[str, str] = ("100_keep", "60_40_spread")


def build_dual_method_comparison(
    data: pd.DataFrame,
    states: pd.Series,
    statistics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build both lagged HMM rotations and both required benchmarks on one Date index."""
    rotations: dict[str, pd.DataFrame] = {}
    for method in ROTATION_METHODS:
        allocation = build_state_allocation(statistics, method)
        rotations[method] = build_rotation_returns(data, states, allocation)

    keep_index = pd.DatetimeIndex(rotations["100_keep"].index, name="Date")
    spread_index = pd.DatetimeIndex(rotations["60_40_spread"].index, name="Date")
    if not keep_index.equals(spread_index):
        raise RuntimeError(
            "Both HMM allocation methods must produce exactly identical return dates."
        )

    equal_weight = build_equal_weight_monthly_returns(data, keep_index)
    spy = build_spy_buy_hold_returns(data, keep_index)
    comparison = pd.DataFrame(
        {
            "hmm_100_keep": rotations["100_keep"]["regime_rotation_return"].to_numpy(dtype=float),
            "hmm_60_40_spread": rotations["60_40_spread"]["regime_rotation_return"].to_numpy(
                dtype=float
            ),
            "equal_weight_monthly": equal_weight.to_numpy(dtype=float),
            "spy_buy_hold": spy.to_numpy(dtype=float),
        },
        index=keep_index,
        columns=list(COMPARISON_COLUMNS),
    )
    comparison.index.name = "Date"
    if not comparison.index.equals(equal_weight.index) or not comparison.index.equals(spy.index):
        raise RuntimeError("Both benchmarks must use exactly the HMM strategy return dates.")
    return comparison, rotations
