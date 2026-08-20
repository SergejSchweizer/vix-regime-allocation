from __future__ import annotations

import pandas as pd
import pytest

from vix_regime_allocation.predictive.dominance import compare_against_assets


def test_positive_and_negative_dominance() -> None:
    idx = pd.date_range("2021-01-01", periods=4, name="Date")
    assets = pd.DataFrame(
        {
            "TLT": [0.001, -0.001, 0.001, 0.0],
            "GLD": [0.002, -0.001, 0.001, 0.0],
            "SPY": [0.003, -0.001, 0.002, -0.001],
        },
        index=idx,
    )
    strategy = pd.Series([0.01, -0.002, 0.009, 0.004], index=idx)
    table, margin, dominates = compare_against_assets(strategy, assets)
    assert list(table["benchmark"]) == ["TLT", "GLD", "SPY"]
    assert margin == pytest.approx(float(table["cagr_difference"].min()))
    assert dominates

    weak = pd.Series([0.0, -0.002, 0.0, -0.001], index=idx)
    _, weak_margin, weak_dominates = compare_against_assets(weak, assets)
    assert weak_margin < 0
    assert not weak_dominates


def test_dominance_requires_identical_dates() -> None:
    idx = pd.date_range("2021-01-01", periods=3, name="Date")
    assets = pd.DataFrame(
        {"TLT": [0.1, 0.0, -0.1], "GLD": [0.0, 0.1, -0.1], "SPY": [0.1, -0.1, 0.0]},
        index=idx,
    )
    with pytest.raises(ValueError):
        compare_against_assets(pd.Series([0.1, 0.0], index=idx[:2]), assets)
