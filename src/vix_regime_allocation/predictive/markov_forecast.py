"""Training-only Markov one-step regime forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import PROBABILITY_TOL, SUPPORTED_STATE_COUNTS


@dataclass(frozen=True)
class MarkovForecastModel:
    """Frozen training-only quantile-state Markov model."""

    n_states: int
    thresholds: np.ndarray
    transition_matrix: np.ndarray
    training_states: pd.Series


def _validate_training(series: pd.Series, n_states: int) -> np.ndarray:
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    if not isinstance(series, pd.Series):
        raise TypeError("training_vix_change must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("training_vix_change index must be a DatetimeIndex.")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("training dates must be unique and sorted ascending.")
    if not is_numeric_dtype(series.dtype):
        raise ValueError("training_vix_change must be numeric.")
    values = series.to_numpy(dtype=float)
    if len(values) < max(2 * n_states, 4) or np.any(~np.isfinite(values)):
        raise ValueError("training_vix_change must contain enough finite observations.")
    return values


def fit_markov_forecaster(training_vix_change: pd.Series, n_states: int) -> MarkovForecastModel:
    """Fit quantile thresholds and a transition matrix from training data only."""

    values = _validate_training(training_vix_change, n_states)
    quantiles = [0.5] if n_states == 2 else [1.0 / 3.0, 2.0 / 3.0]
    thresholds = np.asarray(np.quantile(values, quantiles, method="linear"), dtype=float)
    if np.any(np.diff(thresholds) <= 0.0):
        raise ValueError("training quantile thresholds must be strictly increasing.")
    states_values = np.searchsorted(thresholds, values, side="right").astype(int)
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (states_values[:-1], states_values[1:]), 1.0)
    outgoing = counts.sum(axis=1)
    if np.any(outgoing == 0.0):
        raise ValueError("each Markov state requires at least one outgoing transition.")
    transition = counts / outgoing[:, None]
    if not np.allclose(transition.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0):
        raise ValueError("transition rows must sum to one.")
    states = pd.Series(
        states_values,
        index=training_vix_change.index.copy(),
        name="state",
        dtype="int64",
    )
    return MarkovForecastModel(n_states, thresholds, transition, states)


def forecast_next_regime(model: MarkovForecastModel, current_vix_change: float) -> np.ndarray:
    """Return P(S[t+1] | observed VIX_change[t]) from the frozen transition row."""

    value = float(current_vix_change)
    if not np.isfinite(value):
        raise ValueError("current_vix_change must be finite.")
    state = int(np.searchsorted(model.thresholds, value, side="right"))
    if state < 0 or state >= model.n_states:
        raise RuntimeError("classified state is outside the model state space.")
    probabilities = np.asarray(model.transition_matrix[state], dtype=float).copy()
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -PROBABILITY_TOL)
        or not np.isclose(probabilities.sum(), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        raise ValueError("forecast probabilities are invalid.")
    return probabilities
