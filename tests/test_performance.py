from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.performance import TRADING_DAYS, cumulative_wealth, performance_metrics


def _returns(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-02", periods=len(values), freq="B", name="Date")
    return pd.Series(values, index=index, name="portfolio", dtype=float)


def test_cumulative_wealth_compounds_simple_returns_from_one() -> None:
    returns = _returns([0.10, -0.05, 0.02])
    wealth = cumulative_wealth(returns)

    assert wealth.name == "wealth"
    assert wealth.index.equals(returns.index)
    np.testing.assert_allclose(wealth.to_numpy(), [1.10, 1.045, 1.0659])


def test_performance_metrics_match_exact_manual_formulas() -> None:
    returns = _returns([0.10, -0.05, 0.02])
    result = performance_metrics(returns)

    values = np.array([0.10, -0.05, 0.02])
    terminal = float(np.prod(1.0 + values))
    std = float(np.std(values, ddof=1))
    expected_drawdown = min(0.0, min(1.10 / 1.10 - 1.0, 1.045 / 1.10 - 1.0, 1.0659 / 1.10 - 1.0))

    assert result["cumulative_return"] == pytest.approx(terminal - 1.0)
    assert result["annualized_return"] == pytest.approx(terminal ** (TRADING_DAYS / 3) - 1.0)
    assert result["annualized_volatility"] == pytest.approx(std * math.sqrt(TRADING_DAYS))
    assert result["sharpe_ratio"] == pytest.approx(
        float(np.mean(values)) / std * math.sqrt(TRADING_DAYS)
    )
    assert result["max_drawdown"] == pytest.approx(expected_drawdown)
    assert result["observations"] == 3


def test_first_period_loss_counts_in_drawdown_peak_from_w0_one() -> None:
    returns = _returns([-0.20, 0.10])
    result = performance_metrics(returns)

    assert result["max_drawdown"] == pytest.approx(-0.20)


def test_performance_rejects_zero_volatility_for_sharpe() -> None:
    with pytest.raises(ValueError, match="volatility is zero"):
        performance_metrics(_returns([0.01, 0.01, 0.01]))


@pytest.mark.parametrize("values", [[-1.0, 0.1], [-1.2, 0.1], [float("nan"), 0.1]])
def test_performance_rejects_invalid_simple_returns(values: list[float]) -> None:
    with pytest.raises(ValueError):
        performance_metrics(_returns(values))


def test_performance_requires_two_observations() -> None:
    with pytest.raises(ValueError):
        performance_metrics(_returns([0.01]))


def test_cumulative_wealth_rejects_empty_and_non_series() -> None:
    empty = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        cumulative_wealth(empty)
    with pytest.raises(TypeError):
        cumulative_wealth([0.01, 0.02])  # type: ignore[arg-type]
