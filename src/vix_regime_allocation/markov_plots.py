"""VIX-level figure annotated by quantile states defined on daily VIX changes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


def _validate_vix(vix: pd.Series) -> None:
    if not isinstance(vix, pd.Series):
        raise TypeError("vix must be a pandas Series.")
    if vix.name != "VIX":
        raise ValueError("vix Series must be named 'VIX'.")
    if not isinstance(vix.index, pd.DatetimeIndex):
        raise ValueError("vix index must be a pandas DatetimeIndex.")
    if vix.index.name != "Date" or vix.index.tz is not None:
        raise ValueError("vix index must be timezone-naive and named 'Date'.")
    if vix.index.has_duplicates or not vix.index.is_monotonic_increasing:
        raise ValueError("vix dates must be unique and sorted ascending.")
    if not is_numeric_dtype(vix.dtype):
        raise ValueError("vix must be numeric.")
    values = vix.to_numpy(dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("vix must contain finite, strictly positive observations.")


def _validate_states(states: pd.Series, vix: pd.Series, n_states: int) -> None:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series.")
    if states.name != "state":
        raise ValueError("states Series must be named 'state'.")
    if not states.index.equals(vix.index):
        raise ValueError("state dates must exactly match VIX dates.")
    if not is_integer_dtype(states.dtype):
        raise ValueError("states must have an integer dtype.")
    values = states.to_numpy(dtype=int)
    if np.any((values < 0) | (values >= n_states)):
        raise ValueError(f"states must use labels 0..{n_states - 1}.")


def plot_markov_vix_states(
    vix: pd.Series, states_2: pd.Series, states_3: pd.Series, output_path: Path
) -> None:
    """Plot VIX level for context while coloring states defined from daily VIX changes."""
    _validate_vix(vix)
    _validate_states(states_2, vix, 2)
    _validate_states(states_3, vix, 3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    try:
        vix_values = vix.to_numpy(dtype=float)
        for axis, states, n_states in zip(axes, (states_2, states_3), (2, 3), strict=True):
            axis.plot(vix.index, vix_values, linewidth=1.0, alpha=0.65, label="VIX level")
            state_values = states.to_numpy(dtype=int)
            for state in range(n_states):
                mask = state_values == state
                axis.scatter(
                    vix.index[mask],
                    vix_values[mask],
                    s=7,
                    alpha=0.5,
                    label=f"ΔVIX state {state}",
                )
            axis.set_title(f"VIX level with quantile states defined on daily ΔVIX (K={n_states})")
            axis.set_ylabel("VIX level")
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", ncol=n_states + 1)
        axes[-1].set_xlabel("Date")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        figure.tight_layout()
        figure.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(figure)
