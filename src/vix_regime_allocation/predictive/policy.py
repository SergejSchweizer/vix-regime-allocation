"""Deterministic allocation, turnover, and transaction-cost policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_ORDER, ONE_WAY_COST_BPS


def choose_asset(
    expected_returns: pd.Series, current_asset: str | None, hurdle_bps: float
) -> str:
    """Choose the highest expected-return asset subject to the switching hurdle."""

    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series.")
    if tuple(expected_returns.index.astype(str)) != ASSET_ORDER:
        raise ValueError("expected_returns index must be exactly TLT, GLD, SPY.")
    values = expected_returns.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("expected_returns must be finite.")
    hurdle = float(hurdle_bps)
    if not np.isfinite(hurdle) or hurdle < 0.0:
        raise ValueError("hurdle_bps must be finite and non-negative.")
    best = float(values.max())
    candidate = next(asset for asset in ASSET_ORDER if float(expected_returns[asset]) == best)
    if current_asset is None:
        return candidate
    if current_asset not in ASSET_ORDER:
        raise ValueError("current_asset must be TLT, GLD, SPY, or None.")
    if candidate == current_asset:
        return current_asset
    advantage = float(expected_returns[candidate] - expected_returns[current_asset])
    return candidate if advantage >= hurdle / 10000.0 else current_asset


def one_hot_weights(asset: str) -> np.ndarray:
    """Return TLT/GLD/SPY one-hot weights for a selected asset."""

    if asset not in ASSET_ORDER:
        raise ValueError("asset must be TLT, GLD, or SPY.")
    return np.array([1.0 if item == asset else 0.0 for item in ASSET_ORDER], dtype=float)


def turnover(previous_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """Compute fixed half-L1 turnover."""

    previous = np.asarray(previous_weights, dtype=float)
    new = np.asarray(new_weights, dtype=float)
    if previous.shape != (3,) or new.shape != (3,):
        raise ValueError("weight vectors must have shape (3,).")
    if np.any(~np.isfinite(previous)) or np.any(~np.isfinite(new)):
        raise ValueError("weights must be finite.")
    if np.any(previous < 0.0) or np.any(new < 0.0):
        raise ValueError("weights must be non-negative.")
    if not np.isclose(new.sum(), 1.0):
        raise ValueError("new weights must sum to one.")
    if not (np.isclose(previous.sum(), 0.0) or np.isclose(previous.sum(), 1.0)):
        raise ValueError("previous weights must sum to zero or one.")
    return float(0.5 * np.abs(new - previous).sum())


def apply_transaction_cost(
    gross_return: float, turnover_value: float, cost_bps: float = ONE_WAY_COST_BPS
) -> float:
    """Subtract the fixed one-way turnover cost from the gross simple return."""

    gross = float(gross_return)
    turn = float(turnover_value)
    cost = float(cost_bps)
    if not np.isfinite(gross) or gross <= -1.0:
        raise ValueError("gross_return must be finite and greater than -1.")
    if not np.isfinite(turn) or turn < 0.0:
        raise ValueError("turnover_value must be finite and non-negative.")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("cost_bps must be finite and non-negative.")
    net = gross - turn * cost / 10000.0
    if not np.isfinite(net) or net <= -1.0:
        raise ValueError("net return must remain finite and greater than -1.")
    return float(net)
