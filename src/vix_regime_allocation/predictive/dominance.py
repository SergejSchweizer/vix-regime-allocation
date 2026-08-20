"""Out-of-sample dominance metrics against the three individual assets."""

from __future__ import annotations

import pandas as pd

from vix_regime_allocation.performance import performance_metrics

from .config import ASSET_ORDER


def compare_against_assets(
    strategy_net: pd.Series, asset_returns: pd.DataFrame
) -> tuple[pd.DataFrame, float, bool]:
    """Compare strategy CAGR with TLT, GLD, and SPY on exactly identical dates."""

    if not isinstance(strategy_net, pd.Series):
        raise TypeError("strategy_net must be a pandas Series.")
    if tuple(asset_returns.columns) != ASSET_ORDER:
        raise ValueError("asset_returns columns must be exactly TLT, GLD, SPY.")
    if not strategy_net.index.equals(asset_returns.index):
        raise ValueError("strategy and asset returns must use exactly identical dates.")
    strategy_cagr = float(performance_metrics(strategy_net)["annualized_return"])
    rows: list[dict[str, float | str]] = []
    benchmark_cagrs: list[float] = []
    for asset in ASSET_ORDER:
        cagr = float(performance_metrics(asset_returns[asset].rename(asset))["annualized_return"])
        benchmark_cagrs.append(cagr)
        rows.append(
            {
                "benchmark": asset,
                "benchmark_cagr": cagr,
                "strategy_net_cagr": strategy_cagr,
                "cagr_difference": strategy_cagr - cagr,
            }
        )
    margin = strategy_cagr - max(benchmark_cagrs)
    return pd.DataFrame(rows), float(margin), bool(margin > 0.0)
