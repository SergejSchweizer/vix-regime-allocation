"""Shared Step 5 cumulative-wealth and performance-metric calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

TRADING_DAYS = 252
PERFORMANCE_KEYS: tuple[str, ...] = (
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)


def _validated_returns(returns: pd.Series, *, minimum_observations: int) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if len(returns) < minimum_observations:
        raise ValueError(f"returns must contain at least {minimum_observations} observations.")
    if not is_numeric_dtype(returns.dtype):
        raise ValueError("returns must use a numeric dtype.")
    if returns.index.has_duplicates:
        raise ValueError("returns index must not contain duplicates.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted ascending.")
    values = returns.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")
    if np.any(values <= -1.0):
        raise ValueError("simple returns must be strictly greater than -1.")
    return returns.astype(float)


def cumulative_wealth(returns: pd.Series) -> pd.Series:
    """Compound simple daily returns from initial wealth W0=1."""
    validated = _validated_returns(returns, minimum_observations=1)
    wealth = (1.0 + validated).cumprod()
    wealth.name = "wealth"
    if np.any(~np.isfinite(wealth.to_numpy(dtype=float))) or np.any(
        wealth.to_numpy(dtype=float) <= 0.0
    ):
        raise ValueError("cumulative wealth must stay finite and positive.")
    return wealth


def performance_metrics(returns: pd.Series) -> dict[str, float | int]:
    """Compute the assignment's five fixed performance metrics with risk-free rate zero."""
    validated = _validated_returns(returns, minimum_observations=2)
    wealth = cumulative_wealth(validated)
    n = int(len(validated))
    terminal_wealth = float(wealth.iloc[-1])

    sample_std = float(validated.std(ddof=1))
    if not math.isfinite(sample_std) or sample_std == 0.0:
        raise ValueError("Sharpe ratio is undefined when sample return volatility is zero.")

    cumulative_return = terminal_wealth - 1.0
    annualized_return = terminal_wealth ** (TRADING_DAYS / n) - 1.0
    annualized_volatility = sample_std * math.sqrt(TRADING_DAYS)
    sharpe_ratio = float(validated.mean()) / sample_std * math.sqrt(TRADING_DAYS)

    wealth_values = wealth.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth_values)))
    drawdowns = wealth_values / peaks[1:] - 1.0
    max_drawdown = min(0.0, float(np.min(drawdowns)))

    result: dict[str, float | int] = {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "observations": n,
    }
    numeric_values = np.array(
        [float(result[key]) for key in PERFORMANCE_KEYS if key != "observations"], dtype=float
    )
    if np.any(~np.isfinite(numeric_values)):
        raise ValueError("computed performance metrics must be finite.")
    return result
