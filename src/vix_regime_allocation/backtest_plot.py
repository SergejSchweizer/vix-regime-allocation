"""Step 5 cumulative-performance comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .performance import cumulative_wealth

COMPARISON_COLUMNS: tuple[str, ...] = (
    "regime_rotation",
    "equal_weight_monthly",
    "spy_buy_hold",
)
DISPLAY_LABELS: dict[str, str] = {
    "regime_rotation": "Regime rotation",
    "equal_weight_monthly": "Equal weight (monthly reset)",
    "spy_buy_hold": "SPY buy and hold",
}


def _validate_comparison(comparison: pd.DataFrame) -> None:
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) != COMPARISON_COLUMNS:
        raise ValueError("comparison columns must match the canonical Step 5 order exactly.")
    if not isinstance(comparison.index, pd.DatetimeIndex):
        raise ValueError("comparison index must be a pandas DatetimeIndex.")
    if comparison.index.name != "Date" or comparison.index.tz is not None:
        raise ValueError("comparison index must be timezone-naive and named 'Date'.")
    if len(comparison) == 0:
        raise ValueError("comparison must contain observations.")
    if comparison.index.has_duplicates or not comparison.index.is_monotonic_increasing:
        raise ValueError("comparison dates must be unique and sorted ascending.")
    for column in COMPARISON_COLUMNS:
        if not is_numeric_dtype(comparison[column].dtype):
            raise ValueError(f"comparison column {column!r} must be numeric.")
    values = comparison.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("comparison simple returns must be finite and greater than -1.")


def plot_cumulative_performance(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot exactly three compounded cumulative-return curves and save the figure."""
    _validate_comparison(comparison)
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))
    try:
        for column in COMPARISON_COLUMNS:
            wealth = cumulative_wealth(comparison[column])
            axis.plot(comparison.index, wealth - 1.0, label=DISPLAY_LABELS[column])
        axis.axhline(0.0, linewidth=0.8)
        axis.set_title("Cumulative Performance Comparison")
        axis.set_xlabel("Date")
        axis.set_ylabel("Cumulative return")
        axis.legend()
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
