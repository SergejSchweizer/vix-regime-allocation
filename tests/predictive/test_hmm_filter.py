from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.predictive.hmm_filter as module
from vix_regime_allocation.predictive.hmm_filter import (
    HMMFilterModel,
    filter_observation,
    filtered_probabilities,
    fit_hmm_filter,
    forecast_next_regime,
)


def _model() -> HMMFilterModel:
    return HMMFilterModel(
        n_states=2,
        start_probabilities=np.array([0.6, 0.4]),
        transition_matrix=np.array([[0.8, 0.2], [0.3, 0.7]]),
        means=np.array([-1.0, 1.0]),
        variances=np.array([1.0, 1.0]),
    )


def test_fit_adapter_uses_canonical_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2020-01-01", periods=4, name="Date")
    series = pd.Series([0.0, 0.1, -0.1, 0.2], index=idx, name="VIX_change")
    fake = SimpleNamespace(
        start_probabilities=(0.6, 0.4),
        transition_matrix=pd.DataFrame([[0.8, 0.2], [0.3, 0.7]]),
        means=pd.Series([-1.0, 1.0]),
        variances=pd.Series([1.0, 1.0]),
    )
    monkeypatch.setattr(module, "fit_gaussian_hmm", lambda values, n_states: fake)
    fitted = fit_hmm_filter(series, 2)
    np.testing.assert_allclose(fitted.transition_matrix, _model().transition_matrix)


def test_filter_and_forecast_are_normalized_and_prefix_invariant() -> None:
    model = _model()
    after = filter_observation(model, np.array([0.6, 0.4]), -0.5)
    assert np.isclose(after.sum(), 1.0)
    forecast = forecast_next_regime(model, after)
    np.testing.assert_allclose(forecast, after @ model.transition_matrix)

    idx = pd.date_range("2020-01-01", periods=3, name="Date")
    short = pd.Series([-0.5, 0.2, 1.2], index=idx, name="VIX_change")
    future_index = pd.DatetimeIndex([pd.Timestamp("2020-01-04")], name="Date")
    long = pd.concat([short, pd.Series([-2.0], index=future_index)])
    short_filtered = filtered_probabilities(model, short)
    long_filtered = filtered_probabilities(model, long)
    pd.testing.assert_frame_equal(short_filtered, long_filtered.iloc[:3], check_freq=False)


def test_filter_validation() -> None:
    model = _model()
    with pytest.raises(ValueError):
        filter_observation(model, np.array([1.0, 1.0]), 0.0)
    with pytest.raises(ValueError):
        filter_observation(model, np.array([0.5, 0.5]), np.nan)
