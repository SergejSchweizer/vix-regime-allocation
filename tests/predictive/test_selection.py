from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.selection import (
    build_validation_summary,
    candidate_grid,
    selected_configuration,
)


def _signals(family: str, n_states: int) -> pd.DataFrame:
    decisions = pd.bdate_range("2015-01-02", periods=5)
    returns = pd.bdate_range("2015-01-05", periods=5)
    return pd.DataFrame(
        {
            "decision_date": decisions,
            "return_date": returns,
            "family": [family] * 5,
            "n_states": [n_states] * 5,
            "expected_TLT": [0.0] * 5,
            "expected_GLD": [0.001] * 5,
            "expected_SPY": [0.01] * 5,
        }
    )


def test_exact_grid_and_deterministic_tie_selection() -> None:
    assert len(candidate_grid()) == 16
    signals = {(family, k): _signals(family, k) for family in ("markov", "hmm") for k in (2, 3)}
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=5), name="Date")
    assets = pd.DataFrame(
        {
            "TLT": [0.001, -0.001, 0.001, -0.001, 0.001],
            "GLD": [0.002, -0.001, 0.001, -0.002, 0.002],
            "SPY": [0.01, -0.005, 0.012, -0.004, 0.008],
        },
        index=idx,
    )
    summary = build_validation_summary(signals, assets)
    assert len(summary) == 16
    assert int(summary["selected"].sum()) == 1
    assert selected_configuration(summary) == ("markov", 2, 0.0)


def test_selection_rejects_non_validation_dates() -> None:
    signals = {(family, k): _signals(family, k) for family in ("markov", "hmm") for k in (2, 3)}
    signals[("markov", 2)] = signals[("markov", 2)].copy()
    signals[("markov", 2)].loc[0, "return_date"] = pd.Timestamp("2021-01-04")
    dates = [
        "2015-01-05",
        "2015-01-06",
        "2015-01-07",
        "2015-01-08",
        "2015-01-09",
        "2021-01-04",
    ]
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    assets = pd.DataFrame(
        np.tile([0.001, 0.002, 0.003], (len(idx), 1)),
        index=idx,
        columns=["TLT", "GLD", "SPY"],
    )
    with pytest.raises(ValueError):
        build_validation_summary(signals, assets)
