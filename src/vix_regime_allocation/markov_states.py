"""Quantile discretization of daily VIX changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .model_config import SUPPORTED_STATE_COUNTS


def _validate_vix_change(vix_change: pd.Series) -> np.ndarray:
    if not isinstance(vix_change, pd.Series):
        raise TypeError("vix_change must be a pandas Series.")
    if vix_change.name != "VIX_change":
        raise ValueError("vix_change Series must be named 'VIX_change'.")
    if not isinstance(vix_change.index, pd.DatetimeIndex):
        raise ValueError("vix_change index must be a pandas DatetimeIndex.")
    if vix_change.index.name != "Date":
        raise ValueError("vix_change index must be named 'Date'.")
    if vix_change.index.tz is not None:
        raise ValueError("vix_change index must be timezone-naive.")
    if vix_change.index.has_duplicates or not vix_change.index.is_monotonic_increasing:
        raise ValueError("vix_change dates must be unique and sorted ascending.")
    if not is_numeric_dtype(vix_change.dtype):
        raise ValueError("vix_change must be numeric.")
    values = vix_change.to_numpy(dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("vix_change must contain finite observations.")
    return values


def discretize_vix_change(vix_change: pd.Series, n_states: int) -> tuple[pd.Series, pd.DataFrame]:
    """Discretize ``VIX_change`` into two or three linear-quantile states."""
    values = _validate_vix_change(vix_change)
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")

    quantiles = [0.5] if n_states == 2 else [1.0 / 3.0, 2.0 / 3.0]
    cuts = np.asarray(np.quantile(values, quantiles, method="linear"), dtype=float)
    if np.any(np.diff(cuts) <= 0.0):
        raise ValueError("Quantile cuts must be strictly increasing; duplicate cuts are invalid.")

    state_values = np.searchsorted(cuts, values, side="right").astype(int)
    states = pd.Series(state_values, index=vix_change.index.copy(), name="state", dtype="int64")

    lower = np.concatenate((np.array([-np.inf]), cuts))
    upper = np.concatenate((cuts, np.array([np.inf])))
    thresholds = pd.DataFrame(
        {
            "state": np.arange(n_states, dtype=int),
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )
    return states, thresholds
