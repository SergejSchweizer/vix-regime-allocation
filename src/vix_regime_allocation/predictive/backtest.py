"""Predictive candidate backtest over precomputed causal signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_ORDER, ONE_WAY_COST_BPS
from .policy import apply_transaction_cost, choose_asset, one_hot_weights, turnover

BACKTEST_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "return_date",
    "family",
    "n_states",
    "switch_hurdle_bps",
    "selected_asset",
    "TLT_weight",
    "GLD_weight",
    "SPY_weight",
    "turnover",
    "gross_return",
    "transaction_cost",
    "net_return",
)


def run_candidate_backtest(
    signals: pd.DataFrame,
    asset_returns: pd.DataFrame,
    hurdle_bps: float,
    cost_bps: float = ONE_WAY_COST_BPS,
) -> pd.DataFrame:
    """Apply the deterministic allocation policy to realized next-row returns."""

    if not isinstance(signals, pd.DataFrame) or len(signals) == 0:
        raise ValueError("signals must be a non-empty DataFrame.")
    required = {
        "decision_date",
        "return_date",
        "family",
        "n_states",
        *(f"expected_{asset}" for asset in ASSET_ORDER),
    }
    missing = sorted(required.difference(signals.columns))
    if missing:
        raise ValueError(f"signals are missing required columns: {missing}.")
    if tuple(asset_returns.columns) != ASSET_ORDER:
        raise ValueError("asset_returns columns must be exactly TLT, GLD, SPY.")
    decisions = pd.DatetimeIndex(pd.to_datetime(signals["decision_date"]))
    returns = pd.DatetimeIndex(pd.to_datetime(signals["return_date"]))
    if decisions.has_duplicates or returns.has_duplicates:
        raise ValueError("decision and return dates must be unique.")
    if not decisions.is_monotonic_increasing or not returns.is_monotonic_increasing:
        raise ValueError("signals must be chronological.")
    if not (decisions < returns).all():
        raise ValueError("every decision_date must precede return_date.")
    if not returns.isin(asset_returns.index).all():
        raise ValueError("every return_date must exist in asset_returns.")

    previous_weights = np.zeros(3, dtype=float)
    current_asset: str | None = None
    rows: list[dict[str, object]] = []

    for row in signals.itertuples(index=False):
        expected = pd.Series(
            [float(getattr(row, f"expected_{asset}")) for asset in ASSET_ORDER],
            index=list(ASSET_ORDER),
            dtype=float,
        )
        selected = choose_asset(expected, current_asset, hurdle_bps)
        weights = one_hot_weights(selected)
        turn = turnover(previous_weights, weights)
        return_date = pd.Timestamp(row.return_date)
        gross = float(asset_returns.at[return_date, selected])
        net = apply_transaction_cost(gross, turn, cost_bps)
        transaction_cost = gross - net
        rows.append(
            {
                "decision_date": pd.Timestamp(row.decision_date),
                "return_date": return_date,
                "family": str(row.family),
                "n_states": int(row.n_states),
                "switch_hurdle_bps": float(hurdle_bps),
                "selected_asset": selected,
                "TLT_weight": float(weights[0]),
                "GLD_weight": float(weights[1]),
                "SPY_weight": float(weights[2]),
                "turnover": turn,
                "gross_return": gross,
                "transaction_cost": transaction_cost,
                "net_return": net,
            }
        )
        previous_weights = weights
        current_asset = selected

    result = pd.DataFrame(rows, columns=list(BACKTEST_COLUMNS))
    numeric = result[
        [
            "TLT_weight",
            "GLD_weight",
            "SPY_weight",
            "turnover",
            "gross_return",
            "transaction_cost",
            "net_return",
        ]
    ].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)) or (result["net_return"] <= -1.0).any():
        raise ValueError("candidate backtest produced invalid numeric values.")
    return result
