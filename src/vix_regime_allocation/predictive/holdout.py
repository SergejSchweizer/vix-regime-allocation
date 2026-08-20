"""Frozen final-holdout evaluation for the validation-selected strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vix_regime_allocation.benchmarks import build_equal_weight_monthly_returns
from vix_regime_allocation.performance import performance_metrics

from .backtest import run_candidate_backtest
from .config import ASSET_ORDER, TEST_START
from .dominance import compare_against_assets
from .returns import asset_simple_returns
from .selection import selected_configuration


@dataclass(frozen=True)
class HoldoutResult:
    """Canonical final-test outputs."""

    daily: pd.DataFrame
    performance: pd.DataFrame
    dominance: pd.DataFrame
    cagr_dominance_margin: float
    dominates_all_individual_assets: bool


def _performance_row(
    name: str, returns: pd.Series, mean_turnover: float, switch_count: int
) -> dict[str, object]:
    metrics = performance_metrics(returns)
    return {
        "portfolio": name,
        **metrics,
        "mean_turnover": float(mean_turnover),
        "switch_count": int(switch_count),
    }


def run_final_holdout(
    data: pd.DataFrame,
    validation_summary: pd.DataFrame,
    signals_by_model: dict[tuple[str, int], pd.DataFrame],
) -> HoldoutResult:
    """Run exactly the frozen validation winner on 2021+ test returns."""

    family, n_states, hurdle = selected_configuration(validation_summary)
    key = (family, n_states)
    if set(signals_by_model) != {key}:
        raise ValueError("final holdout must receive signals for the selected model only.")
    signals = signals_by_model[key]
    return_dates = pd.DatetimeIndex(pd.to_datetime(signals["return_date"]), name="Date")
    if len(return_dates) < 2 or (return_dates < TEST_START).any():
        raise ValueError("final holdout returns must lie strictly in the 2021+ test period.")

    asset_returns = asset_simple_returns(data)
    daily = run_candidate_backtest(signals, asset_returns, hurdle)
    if not (
        (daily["family"] == family)
        & (daily["n_states"].astype(int) == n_states)
        & (daily["switch_hurdle_bps"].astype(float) == hurdle)
    ).all():
        raise RuntimeError("final holdout configuration differs from validation winner.")

    index = pd.DatetimeIndex(pd.to_datetime(daily["return_date"]), name="Date")
    gross = pd.Series(daily["gross_return"].to_numpy(dtype=float), index=index, name="strategy_gross")
    net = pd.Series(daily["net_return"].to_numpy(dtype=float), index=index, name="strategy_net")
    selected_assets = daily["selected_asset"].astype(str)
    switches = max(0, int(selected_assets.ne(selected_assets.shift(1)).sum()) - 1)
    mean_turnover = float(daily["turnover"].mean())

    rows = [
        _performance_row("selected_predictive_gross", gross, mean_turnover, switches),
        _performance_row("selected_predictive_net", net, mean_turnover, switches),
    ]
    for asset in ASSET_ORDER:
        rows.append(_performance_row(asset, asset_returns.loc[index, asset].rename(asset), 0.0, 0))
    equal = build_equal_weight_monthly_returns(data, index)
    rows.append(_performance_row("equal_weight_monthly", equal, 0.0, 0))
    performance = pd.DataFrame(rows)

    aligned_assets = asset_returns.loc[index, list(ASSET_ORDER)].copy()
    dominance, margin, dominates = compare_against_assets(net, aligned_assets)
    if np.any(~np.isfinite(performance.select_dtypes(include=[np.number]).to_numpy(dtype=float))):
        raise ValueError("holdout performance must be finite.")
    return HoldoutResult(daily, performance, dominance, margin, dominates)
