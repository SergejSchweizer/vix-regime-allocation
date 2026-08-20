from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.markov_forecast import (
    fit_markov_forecaster,
    forecast_next_regime,
)


def _series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=8, freq="D", name="Date")
    return pd.Series([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], index=idx, name="VIX_change")


def test_markov_fit_and_forecast_are_training_only() -> None:
    model = fit_markov_forecaster(_series(), 2)
    np.testing.assert_allclose(model.thresholds, [0.5])
    assert model.training_states.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    np.testing.assert_allclose(model.transition_matrix, [[0.75, 0.25], [0.0, 1.0]])
    np.testing.assert_allclose(forecast_next_regime(model, -5.0), [0.75, 0.25])
    np.testing.assert_allclose(forecast_next_regime(model, 5.0), [0.0, 1.0])


def test_markov_three_states_and_errors() -> None:
    model = fit_markov_forecaster(_series(), 3)
    assert model.transition_matrix.shape == (3, 3)
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        fit_markov_forecaster(_series(), 4)
    with pytest.raises(ValueError):
        forecast_next_regime(model, np.nan)
