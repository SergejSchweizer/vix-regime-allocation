"""Deterministic Gaussian-HMM fitting for VIX-change regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from pandas.api.types import is_numeric_dtype

from .model_config import (
    HMM_MIN_COVAR,
    HMM_N_ITER,
    HMM_SEEDS,
    HMM_TOL,
    LIKELIHOOD_TIE_TOL,
    PROBABILITY_TOL,
    SUPPORTED_STATE_COUNTS,
)


@dataclass(frozen=True)
class HMMFitResult:
    """Canonical relabeled output of one selected Gaussian-HMM restart."""

    n_states: int
    seed: int
    converged: bool
    log_likelihood: float
    start_probabilities: tuple[float, ...]
    transition_matrix: pd.DataFrame
    means: pd.Series
    variances: pd.Series
    states: pd.Series
    probabilities: pd.DataFrame


def _validated_vix_change(vix_change: pd.Series, n_states: int) -> np.ndarray:
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    if not isinstance(vix_change, pd.Series):
        raise TypeError("vix_change must be a pandas Series.")
    if vix_change.name != "VIX_change":
        raise ValueError("vix_change Series must be named 'VIX_change'.")
    if not isinstance(vix_change.index, pd.DatetimeIndex):
        raise ValueError("vix_change index must be a pandas DatetimeIndex.")
    if vix_change.index.name != "Date" or vix_change.index.tz is not None:
        raise ValueError("vix_change dates must be timezone-naive and named 'Date'.")
    if vix_change.index.has_duplicates or not vix_change.index.is_monotonic_increasing:
        raise ValueError("vix_change dates must be unique and sorted ascending.")
    if not is_numeric_dtype(vix_change.dtype):
        raise ValueError("vix_change must be numeric.")
    values = vix_change.to_numpy(dtype=float)
    if len(values) < max(2 * n_states, 4) or np.any(~np.isfinite(values)):
        raise ValueError("vix_change must contain enough finite observations for the requested K.")
    return values.reshape(-1, 1)


def _fit_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[GaussianHMM, float]:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=HMM_N_ITER,
        tol=HMM_TOL,
        min_covar=HMM_MIN_COVAR,
        random_state=seed,
    )
    model.fit(observations)
    score = float(model.score(observations))
    return model, score


def _select_restart(observations: np.ndarray, n_states: int) -> tuple[GaussianHMM, int, float]:
    """Select the best converged restart without letting one failed seed abort the fit.

    Numerical failures are local to one initialization. The deterministic restart policy
    therefore evaluates every configured seed that can be fitted, ignores failed or
    non-converged/non-finite restarts, then selects the highest finite log-likelihood.
    Likelihood ties within the configured tolerance are resolved by the smallest seed.
    """
    successful: list[tuple[float, int, GaussianHMM]] = []
    for seed in HMM_SEEDS:
        try:
            model, score = _fit_restart(observations, n_states, seed)
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            continue
        if bool(model.monitor_.converged) and np.isfinite(score):
            successful.append((score, seed, model))
    if not successful:
        raise RuntimeError("No configured HMM restart converged with a finite likelihood.")

    best_score = max(item[0] for item in successful)
    tied = [item for item in successful if best_score - item[0] <= LIKELIHOOD_TIE_TOL]
    score, seed, model = min(tied, key=lambda item: item[1])
    return model, seed, score


def _relabel_result(
    model: GaussianHMM,
    seed: int,
    score: float,
    observations: np.ndarray,
    vix_change: pd.Series,
) -> HMMFitResult:
    n_states = model.n_components
    original_means = np.asarray(model.means_, dtype=float).reshape(n_states)
    original_variances = np.asarray(model.covars_, dtype=float).reshape(n_states, -1)[:, 0]
    original_indices = np.arange(n_states)
    order = np.lexsort((original_indices, original_means))

    old_to_new = np.empty(n_states, dtype=int)
    old_to_new[order] = np.arange(n_states)
    original_states = np.asarray(model.predict(observations), dtype=int)
    states = old_to_new[original_states]
    original_probabilities = np.asarray(model.predict_proba(observations), dtype=float)
    probabilities = original_probabilities[:, order]

    start = np.asarray(model.startprob_, dtype=float)[order]
    transition = np.asarray(model.transmat_, dtype=float)[np.ix_(order, order)]
    means = original_means[order]
    variances = original_variances[order]

    if np.any(~np.isfinite(start)) or np.any(start < -PROBABILITY_TOL):
        raise ValueError("HMM start probabilities are invalid.")
    if not np.isclose(start.sum(), 1.0, atol=PROBABILITY_TOL, rtol=0.0):
        raise ValueError("HMM start probabilities must sum to one.")
    if np.any(~np.isfinite(transition)) or np.any(transition < -PROBABILITY_TOL):
        raise ValueError("HMM transition probabilities are invalid.")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0):
        raise ValueError("HMM transition rows must sum to one.")
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("HMM variances must be finite and strictly positive.")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < -PROBABILITY_TOL):
        raise ValueError("HMM posterior probabilities are invalid.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0):
        raise ValueError("HMM posterior rows must sum to one.")

    labels = [f"state_{state}" for state in range(n_states)]
    transition_frame = pd.DataFrame(
        transition,
        index=pd.Index(range(n_states), name="from_state"),
        columns=labels,
    )
    state_index = pd.Index(range(n_states), name="state")
    return HMMFitResult(
        n_states=n_states,
        seed=seed,
        converged=True,
        log_likelihood=score,
        start_probabilities=tuple(float(value) for value in start),
        transition_matrix=transition_frame,
        means=pd.Series(means, index=state_index, name="mean_vix_change"),
        variances=pd.Series(variances, index=state_index, name="variance_vix_change"),
        states=pd.Series(states, index=vix_change.index.copy(), name="state", dtype="int64"),
        probabilities=pd.DataFrame(probabilities, index=vix_change.index.copy(), columns=labels),
    )


def fit_gaussian_hmm(vix_change: pd.Series, n_states: int) -> HMMFitResult:
    """Fit all configured restarts and return the deterministic best converged HMM."""
    observations = _validated_vix_change(vix_change, n_states)
    model, seed, score = _select_restart(observations, n_states)
    return _relabel_result(model, seed, score, observations, vix_change)
