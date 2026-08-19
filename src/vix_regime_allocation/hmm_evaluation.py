"""Candidate assembly for deterministic Gaussian HMM regimes."""

from __future__ import annotations

import pandas as pd

from .hmm_model import HMMFitResult, fit_gaussian_hmm
from .information_criteria import aic, bic, hmm_parameter_count


def evaluate_hmm_candidate(vix_change: pd.Series, n_states: int) -> dict[str, object]:
    """Fit and evaluate one Gaussian-HMM candidate exactly once."""
    fit: HMMFitResult = fit_gaussian_hmm(vix_change, n_states)
    n_parameters = hmm_parameter_count(n_states)
    n_observations = len(vix_change)
    return {
        "family": "hmm",
        "n_states": n_states,
        "log_likelihood": fit.log_likelihood,
        "n_parameters": n_parameters,
        "n_observations": n_observations,
        "aic": aic(fit.log_likelihood, n_parameters),
        "bic": bic(fit.log_likelihood, n_parameters, n_observations),
        "converged": fit.converged,
        "fit": fit,
    }
