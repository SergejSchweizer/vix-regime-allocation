"""Validation-only HMM candidate grid and deterministic predictive-strategy selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vix_regime_allocation.performance import performance_metrics

from .backtest import run_candidate_backtest
from .config import (
    PREDICTIVE_FAMILY,
    SELECTION_TOL,
    SUPPORTED_STATE_COUNTS,
    SWITCH_HURDLES_BPS,
    VALIDATION_END,
    VALIDATION_START,
)

VALIDATION_COLUMNS: tuple[str, ...] = (
    "family",
    "n_states",
    "switch_hurdle_bps",
    "mean_log_growth",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "mean_turnover",
    "switch_count",
    "observations",
    "selected",
)


def candidate_grid() -> tuple[tuple[str, int, float], ...]:
    """Return the exact HMM-only 8-candidate validation grid."""
    return tuple(
        (PREDICTIVE_FAMILY, n_states, hurdle)
        for n_states in SUPPORTED_STATE_COUNTS
        for hurdle in SWITCH_HURDLES_BPS
    )


def _switch_count(selected_assets: pd.Series) -> int:
    if len(selected_assets) == 0:
        return 0
    changes = selected_assets.astype(str).ne(selected_assets.astype(str).shift(1))
    return max(0, int(changes.sum()) - 1)


def _select_index(summary: pd.DataFrame) -> int:
    candidates = summary.copy()
    if not (candidates["family"].astype(str) == PREDICTIVE_FAMILY).all():
        raise ValueError("Predictive validation supports HMM candidates only.")
    best_growth = float(candidates["mean_log_growth"].max())
    candidates = candidates.loc[
        candidates["mean_log_growth"].astype(float).sub(best_growth).abs() <= SELECTION_TOL
    ]
    best_turnover = float(candidates["mean_turnover"].min())
    candidates = candidates.loc[
        candidates["mean_turnover"].astype(float).sub(best_turnover).abs() <= SELECTION_TOL
    ]
    minimum_states = int(candidates["n_states"].min())
    candidates = candidates.loc[candidates["n_states"].astype(int) == minimum_states]
    minimum_hurdle = float(candidates["switch_hurdle_bps"].min())
    candidates = candidates.loc[
        candidates["switch_hurdle_bps"].astype(float) == minimum_hurdle
    ]
    return int(candidates.index[0])


def build_validation_summary(
    signals_by_model: dict[tuple[str, int], pd.DataFrame],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the fixed HMM K=2/K=3 candidates on validation returns only."""
    expected_model_keys = {(PREDICTIVE_FAMILY, n_states) for n_states in SUPPORTED_STATE_COUNTS}
    if set(signals_by_model) != expected_model_keys:
        raise ValueError("signals_by_model must contain HMM K=2 and HMM K=3 exactly.")

    rows: list[dict[str, object]] = []
    for family, n_states, hurdle in candidate_grid():
        signals = signals_by_model[(family, n_states)]
        if "family" not in signals.columns or not (signals["family"].astype(str) == "hmm").all():
            raise ValueError("Predictive signal frames must contain family='hmm' only.")
        if "n_states" not in signals.columns or not (
            signals["n_states"].astype(int) == n_states
        ).all():
            raise ValueError("Predictive signal state count does not match its model key.")
        return_dates = pd.DatetimeIndex(pd.to_datetime(signals["return_date"]))
        if len(return_dates) < 2:
            raise ValueError("each validation candidate requires at least two return rows.")
        if (return_dates < VALIDATION_START).any() or (return_dates > VALIDATION_END).any():
            raise ValueError("validation selection cannot consume dates outside 2015-2020.")
        backtest = run_candidate_backtest(signals, asset_returns, hurdle)
        net = pd.Series(
            backtest["net_return"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(backtest["return_date"]), name="Date"),
            name="net_return",
        )
        metrics = performance_metrics(net)
        mean_log_growth = float(np.log1p(net.to_numpy(dtype=float)).mean())
        rows.append(
            {
                "family": PREDICTIVE_FAMILY,
                "n_states": n_states,
                "switch_hurdle_bps": hurdle,
                "mean_log_growth": mean_log_growth,
                "cumulative_return": float(metrics["cumulative_return"]),
                "annualized_return": float(metrics["annualized_return"]),
                "annualized_volatility": float(metrics["annualized_volatility"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "mean_turnover": float(backtest["turnover"].mean()),
                "switch_count": _switch_count(backtest["selected_asset"]),
                "observations": int(metrics["observations"]),
                "selected": False,
            }
        )
    summary = pd.DataFrame(rows, columns=list(VALIDATION_COLUMNS))
    if len(summary) != len(SUPPORTED_STATE_COUNTS) * len(SWITCH_HURDLES_BPS):
        raise RuntimeError("HMM-only validation candidate grid must contain exactly 8 rows.")
    winner = _select_index(summary)
    summary.loc[winner, "selected"] = True
    if int(summary["selected"].sum()) != 1:
        raise RuntimeError("validation must select exactly one candidate.")
    return summary


def selected_configuration(summary: pd.DataFrame) -> tuple[str, int, float]:
    """Return the single frozen HMM validation winner."""
    if not isinstance(summary, pd.DataFrame) or "selected" not in summary.columns:
        raise ValueError("summary must contain a selected column.")
    if "family" not in summary.columns or not (
        summary["family"].astype(str) == PREDICTIVE_FAMILY
    ).all():
        raise ValueError("summary must contain HMM candidates only.")
    selected = summary.loc[summary["selected"].astype(bool)]
    if len(selected) != 1:
        raise ValueError("summary must contain exactly one selected candidate.")
    row = selected.iloc[0]
    return PREDICTIVE_FAMILY, int(row["n_states"]), float(row["switch_hurdle_bps"])
