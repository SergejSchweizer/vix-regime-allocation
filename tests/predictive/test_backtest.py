from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.backtest import run_candidate_backtest


def _signals() -> pd.DataFrame:
    decisions = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    returns = pd.to_datetime(["2021-01-05", "2021-01-06", "2021-01-07"])
    return pd.DataFrame(
        {
            "decision_date": decisions,
            "return_date": returns,
            "family": ["markov"] * 3,
            "n_states": [2] * 3,
            "expected_TLT": [0.02, 0.00, 0.03],
            "expected_GLD": [0.01, 0.01, 0.01],
            "expected_SPY": [0.00, 0.02, 0.00],
        }
    )


def test_candidate_backtest_switches_and_costs() -> None:
    idx = pd.DatetimeIndex(pd.to_datetime(["2021-01-05", "2021-01-06", "2021-01-07"]), name="Date")
    returns = pd.DataFrame(
        {"TLT": [0.01, 0.0, 0.03], "GLD": [0.0, 0.0, 0.0], "SPY": [0.0, 0.02, 0.0]},
        index=idx,
    )
    result = run_candidate_backtest(_signals(), returns, hurdle_bps=0.0, cost_bps=5.0)
    assert result["selected_asset"].tolist() == ["TLT", "SPY", "TLT"]
    np.testing.assert_allclose(result["turnover"], [0.5, 1.0, 1.0])
    np.testing.assert_allclose(result["transaction_cost"], [0.00025, 0.0005, 0.0005])
    np.testing.assert_allclose(result["net_return"], [0.00975, 0.0195, 0.0295])
    expected_wealth = np.prod(1.0 + np.array([0.00975, 0.0195, 0.0295]))
    assert np.prod(1.0 + result["net_return"].to_numpy()) == pytest.approx(expected_wealth)


def test_candidate_backtest_rejects_missing_return_date() -> None:
    returns = pd.DataFrame(
        {"TLT": [0.01], "GLD": [0.0], "SPY": [0.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2021-01-05")], name="Date"),
    )
    with pytest.raises(ValueError):
        run_candidate_backtest(_signals(), returns, 0.0)
