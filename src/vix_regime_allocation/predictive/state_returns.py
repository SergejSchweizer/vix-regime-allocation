"""Training-only state-conditioned ETF return expectations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_ORDER, PROBABILITY_TOL


def _validate_returns(simple_returns: pd.DataFrame) -> None:
    if not isinstance(simple_returns, pd.DataFrame):
        raise TypeError("simple_returns must be a pandas DataFrame.")
    if tuple(simple_returns.columns) != ASSET_ORDER:
        raise ValueError("simple_returns columns must be exactly TLT, GLD, SPY.")
    values = simple_returns.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("simple_returns must be finite and greater than -1.")


def hard_state_asset_means(
    simple_returns: pd.DataFrame, states: pd.Series, n_states: int
) -> pd.DataFrame:
    """Compute hard-state conditional simple-return means from a training prefix."""

    _validate_returns(simple_returns)
    if not isinstance(states, pd.Series) or not states.index.equals(simple_returns.index):
        raise ValueError("states must be a Series on exactly the return index.")
    values = states.to_numpy(dtype=int)
    if not np.array_equal(np.unique(values), np.arange(n_states)):
        raise ValueError("states must contain every contiguous state 0..K-1.")
    rows: list[np.ndarray] = []
    for state in range(n_states):
        mask = values == state
        if not np.any(mask):
            raise ValueError("each state requires observations.")
        rows.append(simple_returns.loc[mask, list(ASSET_ORDER)].mean().to_numpy(dtype=float))
    result = pd.DataFrame(rows, index=pd.Index(range(n_states), name="state"), columns=ASSET_ORDER)
    if np.any(~np.isfinite(result.to_numpy(dtype=float))):
        raise ValueError("hard-state means must be finite.")
    return result


def soft_state_asset_means(
    simple_returns: pd.DataFrame, filtered_probabilities: pd.DataFrame
) -> pd.DataFrame:
    """Compute HMM state means using one-sided filtered probabilities as soft weights."""

    _validate_returns(simple_returns)
    if not isinstance(filtered_probabilities, pd.DataFrame):
        raise TypeError("filtered_probabilities must be a DataFrame.")
    if not filtered_probabilities.index.equals(simple_returns.index):
        raise ValueError("filtered probabilities must use exactly the return index.")
    n_states = filtered_probabilities.shape[1]
    expected_columns = [f"state_{state}" for state in range(n_states)]
    if list(filtered_probabilities.columns) != expected_columns:
        raise ValueError("filtered probability columns are invalid.")
    probabilities = filtered_probabilities.to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -PROBABILITY_TOL)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        raise ValueError("filtered probabilities must be finite and normalized.")
    returns = simple_returns.to_numpy(dtype=float)
    masses = probabilities.sum(axis=0)
    if np.any(masses <= 0.0):
        raise ValueError("each HMM state requires positive filtered probability mass.")
    means = probabilities.T @ returns / masses[:, None]
    result = pd.DataFrame(means, index=pd.Index(range(n_states), name="state"), columns=ASSET_ORDER)
    if np.any(~np.isfinite(result.to_numpy(dtype=float))):
        raise ValueError("soft-state means must be finite.")
    return result


def expected_asset_returns(
    next_state_probabilities: np.ndarray, state_asset_means: pd.DataFrame
) -> pd.Series:
    """Map a forecast regime distribution into expected next-row asset returns."""

    if not isinstance(state_asset_means, pd.DataFrame):
        raise TypeError("state_asset_means must be a DataFrame.")
    if tuple(state_asset_means.columns) != ASSET_ORDER:
        raise ValueError("state_asset_means columns must be exactly TLT, GLD, SPY.")
    probabilities = np.asarray(next_state_probabilities, dtype=float)
    if probabilities.shape != (len(state_asset_means),):
        raise ValueError("next-state probability shape does not match state means.")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -PROBABILITY_TOL)
        or not np.isclose(probabilities.sum(), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        raise ValueError("next-state probabilities must be finite and normalized.")
    means = state_asset_means.to_numpy(dtype=float)
    if np.any(~np.isfinite(means)):
        raise ValueError("state_asset_means must be finite.")
    forecast = probabilities @ means
    return pd.Series(forecast, index=list(ASSET_ORDER), name="expected_return", dtype=float)
