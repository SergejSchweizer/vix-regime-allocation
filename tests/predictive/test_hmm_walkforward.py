from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.predictive.hmm_walkforward as module
from vix_regime_allocation.predictive.hmm_filter import HMMFilterModel
from vix_regime_allocation.predictive.hmm_walkforward import build_hmm_signals


def _data() -> pd.DataFrame:
    idx = pd.bdate_range("2020-12-15", periods=15, name="Date")
    return pd.DataFrame(
        {
            "TLT_log_return": np.linspace(-0.01, 0.01, len(idx)),
            "GLD_log_return": np.linspace(0.005, -0.005, len(idx)),
            "SPY_log_return": np.linspace(-0.02, 0.02, len(idx)),
            "VIX_change": np.linspace(-1.0, 1.0, len(idx)),
        },
        index=idx,
    )


def test_hmm_walkforward_consumes_current_observation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HMMFilterModel(
        2,
        np.array([0.5, 0.5]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
        np.array([-1.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    monkeypatch.setattr(module, "fit_hmm_filter", lambda values, n_states: model)
    monkeypatch.setattr(
        module,
        "filtered_probabilities",
        lambda fitted, values: pd.DataFrame(
            {"state_0": np.full(len(values), 0.5), "state_1": np.full(len(values), 0.5)},
            index=values.index,
        ),
    )

    observed: list[float] = []

    def fake_filter(fitted: HMMFilterModel, alpha: np.ndarray, observation: float) -> np.ndarray:
        observed.append(observation)
        return np.array([0.6, 0.4])

    monkeypatch.setattr(module, "filter_observation", fake_filter)
    monkeypatch.setattr(
        module, "forecast_next_regime", lambda fitted, alpha: np.array([0.55, 0.45])
    )

    data = _data()
    decisions = data.index[8:12]
    result = build_hmm_signals(data, decisions, 2)
    assert observed == data.loc[decisions, "VIX_change"].tolist()
    assert (result["training_end"] < result["decision_date"]).all()
    assert result["return_date"].tolist() == data.index[9:13].tolist()
    assert np.allclose(result[["p_state_0", "p_state_1"]].sum(axis=1), 1.0)
