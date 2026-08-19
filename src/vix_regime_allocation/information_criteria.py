"""Information-criterion helpers for candidate regime models."""

from __future__ import annotations

import math

from .model_config import SUPPORTED_STATE_COUNTS


def _validate_log_likelihood(log_likelihood: float) -> float:
    value = float(log_likelihood)
    if not math.isfinite(value):
        raise ValueError("log_likelihood must be finite.")
    return value


def _validate_n_parameters(n_parameters: int) -> int:
    if isinstance(n_parameters, bool) or not isinstance(n_parameters, int) or n_parameters < 0:
        raise ValueError("n_parameters must be a non-negative integer.")
    return n_parameters


def aic(log_likelihood: float, n_parameters: int) -> float:
    """Return Akaike's information criterion ``2k - 2 logL``."""
    log_l = _validate_log_likelihood(log_likelihood)
    k = _validate_n_parameters(n_parameters)
    return 2.0 * k - 2.0 * log_l


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Return Schwarz's Bayesian information criterion ``k ln(n) - 2 logL``."""
    log_l = _validate_log_likelihood(log_likelihood)
    k = _validate_n_parameters(n_parameters)
    if isinstance(n_observations, bool) or not isinstance(n_observations, int):
        raise ValueError("n_observations must be a positive integer.")
    if n_observations <= 0:
        raise ValueError("n_observations must be a positive integer.")
    return k * math.log(n_observations) - 2.0 * log_l


def markov_parameter_count(n_states: int) -> int:
    """Free transition-probability count for a K-state Markov chain."""
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    return n_states * (n_states - 1)


def hmm_parameter_count(n_states: int) -> int:
    """Parameter count for a univariate Gaussian HMM with K states."""
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    return n_states**2 + 2 * n_states - 1
