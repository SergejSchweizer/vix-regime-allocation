from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.backtest_plot as module
from vix_regime_allocation.backtest_plot import plot_four_portfolio_cumulative_performance
from vix_regime_allocation.strategy_comparison import COMPARISON_COLUMNS


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
        columns=list(COMPARISON_COLUMNS),
    )


def test_four_curve_contract_has_exact_labels_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    original = module.cumulative_wealth

    def recording_wealth(series: pd.Series) -> pd.Series:
        seen.append(str(series.name))
        return original(series)

    monkeypatch.setattr(module, "cumulative_wealth", recording_wealth)
    output = tmp_path / "four.png"
    plot_four_portfolio_cumulative_performance(_comparison(), output)

    assert seen == list(COMPARISON_COLUMNS)
    assert output.is_file() and output.stat().st_size > 0
    assert plt.get_fignums() == []


def test_four_curve_compounding_matches_manual_terminal_values() -> None:
    comparison = _comparison()
    for column in COMPARISON_COLUMNS:
        expected = float(np.prod(1.0 + comparison[column].to_numpy(dtype=float)) - 1.0)
        actual = float(module.cumulative_wealth(comparison[column]).iloc[-1] - 1.0)
        assert actual == pytest.approx(expected)


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
def test_legacy_or_malformed_comparison_is_rejected(mutator, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((TypeError, ValueError)):
        plot_four_portfolio_cumulative_performance(mutator(_comparison()), tmp_path / "bad.png")
