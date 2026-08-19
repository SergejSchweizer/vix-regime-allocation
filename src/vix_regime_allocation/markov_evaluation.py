"""Candidate assembly for quantile-discretized Markov regimes."""

from __future__ import annotations

import math

import pandas as pd

from .information_criteria import aic, bic, markov_parameter_count
from .markov_chain import estimate_transition_matrix, stationary_distribution
from .markov_states import discretize_vix_change


def _conditional_log_likelihood(states: pd.Series, transition: pd.DataFrame) -> float:
    values = states.to_numpy(dtype=int)
    matrix = transition.to_numpy(dtype=float)
    total = 0.0
    for current, following in zip(values[:-1], values[1:], strict=True):
        probability = float(matrix[current, following])
        if not math.isfinite(probability) or probability <= 0.0:
            raise ValueError("Every observed transition must have a finite positive probability.")
        total += math.log(probability)
    return total


def evaluate_markov_candidate(vix_change: pd.Series, n_states: int) -> dict[str, object]:
    """Build one complete Markov candidate from shared project helpers."""
    states, thresholds = discretize_vix_change(vix_change, n_states)
    transition = estimate_transition_matrix(states, n_states)
    stationary = stationary_distribution(transition)
    log_likelihood = _conditional_log_likelihood(states, transition)
    n_parameters = markov_parameter_count(n_states)
    n_observations = len(states) - 1
    return {
        "family": "markov",
        "n_states": n_states,
        "log_likelihood": log_likelihood,
        "n_parameters": n_parameters,
        "n_observations": n_observations,
        "aic": aic(log_likelihood, n_parameters),
        "bic": bic(log_likelihood, n_parameters, n_observations),
        "converged": True,
        "thresholds": thresholds,
        "transition": transition,
        "stationary": stationary,
        "states": states,
    }
