from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.backtest_summary as module
from vix_regime_allocation.backtest_summary import (
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    build_four_portfolio_performance_summary,
    build_performance_summary,
)
from vix_regime_allocation.performance import PERFORMANCE_KEYS
from vix_regime_allocation.strategy_comparison import COMPARISON_COLUMNS as DUAL_COLUMNS


def _comparison() -> pd.DataFrame:
    index = pd.date_range("2026-01-02", periods=4, freq="B", name="Date")
    return pd.DataFrame(
        {
            "hmm_100_keep": [0.01, -0.02, 0.03, 0.01],
            "hmm_60_40_spread": [0.008, -0.01, 0.02, 0.005],
            "equal_weight_monthly": [0.004, -0.005, 0.01, 0.002],
            "spy_buy_hold": [0.012, -0.03, 0.025, 0.015],
        },
        index=index,
        columns=list(DUAL_COLUMNS),
    )


def test_four_portfolio_summary_has_exact_schema_and_order() -> None:
    summary = build_four_portfolio_performance_summary(_comparison())
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary["portfolio"].tolist() == list(DUAL_COLUMNS)
    assert summary["observations"].tolist() == [4, 4, 4, 4]
    assert np.isfinite(summary.drop(columns="portfolio").to_numpy(dtype=float)).all()


def test_four_portfolio_summary_delegates_each_series_to_shared_metric_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_metrics(series: pd.Series) -> dict[str, float | int]:
        seen.append(str(series.name))
        offset = float(len(seen))
        return {
            "cumulative_return": offset,
            "annualized_return": offset + 1,
            "annualized_volatility": offset + 2,
            "sharpe_ratio": offset + 3,
            "max_drawdown": -offset,
            "observations": len(series),
        }

    monkeypatch.setattr(module, "performance_metrics", fake_metrics)
    summary = build_four_portfolio_performance_summary(_comparison())
    assert seen == list(DUAL_COLUMNS)
    assert summary["portfolio"].tolist() == list(DUAL_COLUMNS)
    assert tuple(fake_metrics(_comparison().iloc[:, 0]).keys()) == PERFORMANCE_KEYS


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns="hmm_60_40_spread"),
        lambda frame: frame.assign(extra=0.0),
        lambda frame: frame.loc[:, list(reversed(frame.columns))],
        lambda frame: frame.rename_axis("date"),
        lambda frame: frame.iloc[::-1],
        lambda frame: frame.assign(hmm_100_keep=np.nan),
    ],
)
def test_invalid_or_legacy_schema_is_rejected_by_strict_four_portfolio_api(mutator) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((TypeError, ValueError)):
        build_four_portfolio_performance_summary(mutator(_comparison()))


def test_dispatcher_uses_four_portfolio_path_for_new_comparison() -> None:
    direct = build_four_portfolio_performance_summary(_comparison())
    dispatched = build_performance_summary(_comparison())
    pd.testing.assert_frame_equal(dispatched, direct)


def test_temporary_legacy_dispatch_remains_scoped_to_exact_old_three_columns() -> None:
    index = _comparison().index
    legacy = pd.DataFrame(
        {
            "regime_rotation": [0.01, 0.02, -0.01, 0.01],
            "equal_weight_monthly": [0.005, 0.006, -0.002, 0.004],
            "spy_buy_hold": [0.012, 0.01, -0.015, 0.008],
        },
        index=index,
        columns=list(COMPARISON_COLUMNS),
    )
    summary = build_performance_summary(legacy)
    assert summary["portfolio"].tolist() == list(COMPARISON_COLUMNS)
    with pytest.raises(ValueError, match="supported Step 5 schema"):
        build_performance_summary(legacy.assign(unexpected=0.0))
