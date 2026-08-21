from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.holdout import run_final_holdout


def _data() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]),
        name="Date",
    )
    simple = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.002, 0.003],
            [0.002, -0.001, 0.01],
            [-0.001, 0.002, -0.004],
            [0.003, 0.001, 0.008],
        ]
    )
    prices = np.full((len(idx), 4), 100.0)
    return pd.DataFrame(
        {
            "TLT": prices[:, 0],
            "GLD": prices[:, 1],
            "SPY": prices[:, 2],
            "VIX": prices[:, 3],
            "TLT_log_return": np.log1p(simple[:, 0]),
            "GLD_log_return": np.log1p(simple[:, 1]),
            "SPY_log_return": np.log1p(simple[:, 2]),
            "VIX_change": np.linspace(-1.0, 1.0, len(idx)),
        },
        index=idx,
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [{"family": "hmm", "n_states": 2, "switch_hurdle_bps": 0.0, "selected": True}]
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
            "return_date": pd.to_datetime(["2021-01-05", "2021-01-06", "2021-01-07"]),
            "family": ["hmm"] * 3,
            "n_states": [2] * 3,
            "expected_TLT": [0.0, 0.0, 0.0],
            "expected_GLD": [0.0, 0.0, 0.0],
            "expected_SPY": [0.01, 0.01, 0.01],
        }
    )


def test_holdout_freezes_selected_configuration() -> None:
    result = run_final_holdout(_data(), _summary(), {("hmm", 2): _signals()})
    assert set(result.performance["portfolio"]) == {
        "selected_predictive_gross",
        "selected_predictive_net",
        "TLT",
        "GLD",
        "SPY",
        "equal_weight_monthly",
    }
    assert list(result.dominance["benchmark"]) == ["TLT", "GLD", "SPY"]
    assert isinstance(result.dominates_all_individual_assets, bool)


def test_holdout_rejects_extra_model() -> None:
    with pytest.raises(ValueError):
        run_final_holdout(
            _data(),
            _summary(),
            {("hmm", 2): _signals(), ("hmm", 3): _signals()},
        )
