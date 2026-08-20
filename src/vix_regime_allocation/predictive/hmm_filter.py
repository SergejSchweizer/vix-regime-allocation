"""One-sided Gaussian-HMM filtering for causal regime forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from vix_regime_allocation.hmm_model import fit_gaussian_hmm

from .config import PROBABILITY_TOL, SUPPORTED_STATE_COUNTS


@dataclass(frozen=True)
class HMMFilterModel:
    """Frozen HMM parameters required for sequential filtering."""

    n_states: int
    start_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def fit_hmm_filter(training_vix_change: pd.Series, n_states: int) -> HMMFilterModel:
    """Fit the deterministic canonical HMM on a training prefix only."""

    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    fit = fit_gaussian_hmm(training_vix_change, n_states)
    return HMMFilterModel(
        n_states=n_states,
        start_probabilities=np.asarray(fit.start_probabilities, dtype=float),
        transition_matrix=fit.transition_matrix.to_numpy(dtype=float),
        means=fit.means.to_numpy(dtype=float),
        variances=fit.variances.to_numpy(dtype=float),
    )


def _validate_probability_vector(values: np.ndarray, n_states: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (n_states,):
        raise ValueError("probability vector has an invalid shape.")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -PROBABILITY_TOL)
        or not np.isclose(probabilities.sum(), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        raise ValueError("probability vector must be finite, non-negative, and normalized.")
    return probabilities


def _posterior_from_prior(
    model: HMMFilterModel, prior: np.ndarray, observation: float
) -> np.ndarray:
    prior_values = _validate_probability_vector(prior, model.n_states)
    value = float(observation)
    if not np.isfinite(value):
        raise ValueError("observation must be finite.")
    log_emission = (
        -0.5 * np.log(2.0 * np.pi * model.variances)
        - 0.5 * ((value - model.means) ** 2) / model.variances
    )
    log_weight = np.log(np.clip(prior_values, np.finfo(float).tiny, None)) + log_emission
    shifted = log_weight - float(np.max(log_weight))
    weights = np.exp(shifted)
    denominator = float(weights.sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("HMM filtering normalization denominator must be positive.")
    posterior = weights / denominator
    return _validate_probability_vector(posterior, model.n_states).copy()


def filter_observation(
    model: HMMFilterModel, prior_filtered: np.ndarray, observation: float
) -> np.ndarray:
    """Advance one step and condition on exactly one newly observed VIX change."""

    previous = _validate_probability_vector(prior_filtered, model.n_states)
    prior = previous @ model.transition_matrix
    return _posterior_from_prior(model, prior, observation)


def forecast_next_regime(model: HMMFilterModel, filtered: np.ndarray) -> np.ndarray:
    """Return one-step-ahead regime probabilities alpha[t] @ P."""

    alpha = _validate_probability_vector(filtered, model.n_states)
    forecast = np.asarray(alpha @ model.transition_matrix, dtype=float)
    return _validate_probability_vector(forecast, model.n_states).copy()


def filtered_probabilities(
    model: HMMFilterModel, observations: pd.Series
) -> pd.DataFrame:
    """Filter a complete training prefix sequentially with no future conditioning."""

    if not isinstance(observations, pd.Series):
        raise TypeError("observations must be a pandas Series.")
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise ValueError("observations index must be a DatetimeIndex.")
    if observations.index.has_duplicates or not observations.index.is_monotonic_increasing:
        raise ValueError("observation dates must be unique and sorted ascending.")
    if not is_numeric_dtype(observations.dtype):
        raise ValueError("observations must be numeric.")
    values = observations.to_numpy(dtype=float)
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("observations must contain finite values.")
    rows: list[np.ndarray] = []
    current = _posterior_from_prior(model, model.start_probabilities, float(values[0]))
    rows.append(current)
    for value in values[1:]:
        current = filter_observation(model, current, float(value))
        rows.append(current)
    columns = [f"state_{state}" for state in range(model.n_states)]
    return pd.DataFrame(rows, index=observations.index.copy(), columns=columns)
