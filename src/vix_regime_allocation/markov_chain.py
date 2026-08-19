"""Transition and stationary-distribution estimators for discrete regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .model_config import STATIONARY_TOL, SUPPORTED_STATE_COUNTS


def _validated_states(states: pd.Series, n_states: int) -> np.ndarray:
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series.")
    if states.name != "state":
        raise ValueError("states Series must be named 'state'.")
    if len(states) < 2:
        raise ValueError("At least two state observations are required.")
    if not is_integer_dtype(states.dtype):
        raise ValueError("states must have an integer dtype.")
    values = states.to_numpy(dtype=int)
    if np.any((values < 0) | (values >= n_states)):
        raise ValueError("states contain labels outside the configured range.")
    return values


def estimate_transition_matrix(states: pd.Series, n_states: int) -> pd.DataFrame:
    """Estimate row-stochastic transition probabilities from consecutive states."""
    values = _validated_states(states, n_states)
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (values[:-1], values[1:]), 1.0)
    outgoing = counts.sum(axis=1)
    if np.any(outgoing == 0.0):
        missing = np.flatnonzero(outgoing == 0.0).tolist()
        raise ValueError(f"Every expected state needs an outgoing transition; missing {missing}.")
    probabilities = counts / outgoing[:, None]
    columns = [f"state_{state}" for state in range(n_states)]
    return pd.DataFrame(
        probabilities,
        index=pd.Index(range(n_states), name="from_state"),
        columns=columns,
    )


def _validated_transition(transition: pd.DataFrame) -> np.ndarray:
    if not isinstance(transition, pd.DataFrame):
        raise TypeError("transition must be a pandas DataFrame.")
    n_states = len(transition)
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError("transition must contain exactly two or three states.")
    expected_columns = [f"state_{state}" for state in range(n_states)]
    if list(transition.columns) != expected_columns:
        raise ValueError(f"transition columns must be exactly {expected_columns}.")
    if list(transition.index) != list(range(n_states)) or transition.index.name != "from_state":
        raise ValueError("transition index must be from_state=0..K-1.")
    matrix = transition.to_numpy(dtype=float)
    if np.any(~np.isfinite(matrix)) or np.any(matrix < -STATIONARY_TOL):
        raise ValueError("transition probabilities must be finite and non-negative.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=STATIONARY_TOL, rtol=0.0):
        raise ValueError("transition rows must sum to one.")
    return matrix


def stationary_distribution(transition: pd.DataFrame) -> pd.Series:
    """Return the unique stationary row distribution of a valid transition matrix."""
    matrix = _validated_transition(transition)
    n_states = len(matrix)
    kernel = matrix.T - np.eye(n_states)
    rank = np.linalg.matrix_rank(kernel, tol=STATIONARY_TOL)
    if n_states - rank != 1:
        raise ValueError("transition does not have a unique stationary distribution.")

    system = np.vstack((kernel, np.ones((1, n_states))))
    target = np.concatenate((np.zeros(n_states), np.array([1.0])))
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    if np.any(solution < -STATIONARY_TOL):
        raise ValueError("stationary solution contains a materially negative probability.")
    solution = np.clip(solution, 0.0, None)
    solution = solution / solution.sum()
    if not np.allclose(solution @ matrix, solution, atol=STATIONARY_TOL, rtol=0.0):
        raise ValueError("stationary solution does not satisfy pi @ P = pi.")
    return pd.Series(
        solution,
        index=pd.Index(range(n_states), name="state"),
        name="stationary_probability",
    )
