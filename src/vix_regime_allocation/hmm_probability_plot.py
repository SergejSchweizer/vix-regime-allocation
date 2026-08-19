"""Smoothed-posterior probability figure for Gaussian-HMM regimes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_config import PROBABILITY_TOL


def _validate_probabilities(probabilities: pd.DataFrame, n_states: int) -> None:
    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("probabilities must be a pandas DataFrame.")
    expected = [f"state_{state}" for state in range(n_states)]
    if list(probabilities.columns) != expected:
        raise ValueError(f"probability columns must be exactly {expected}.")
    if not isinstance(probabilities.index, pd.DatetimeIndex):
        raise ValueError("probability index must be a pandas DatetimeIndex.")
    if probabilities.index.name != "Date" or probabilities.index.tz is not None:
        raise ValueError("probability index must be timezone-naive and named 'Date'.")
    if probabilities.index.has_duplicates or not probabilities.index.is_monotonic_increasing:
        raise ValueError("probability dates must be unique and sorted ascending.")
    values = probabilities.to_numpy(dtype=float)
    if values.shape[0] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("probabilities must contain finite observations.")
    if np.any(values < -PROBABILITY_TOL) or np.any(values > 1.0 + PROBABILITY_TOL):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0):
        raise ValueError("posterior probability rows must sum to one.")


def plot_hmm_smoothed_probabilities(
    probabilities_2: pd.DataFrame, probabilities_3: pd.DataFrame, output_path: Path
) -> None:
    """Write the canonical two-panel smoothed-posterior figure."""
    _validate_probabilities(probabilities_2, 2)
    _validate_probabilities(probabilities_3, 3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for axis, probabilities, n_states in zip(
        axes, (probabilities_2, probabilities_3), (2, 3), strict=True
    ):
        for column in probabilities.columns:
            axis.plot(
                probabilities.index,
                probabilities[column].to_numpy(dtype=float),
                label=column.replace("_", " ").title(),
            )
        axis.set_title(f"Gaussian HMM smoothed state probabilities (K={n_states})")
        axis.set_ylabel("Posterior probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=n_states)
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
