"""Step 5 cumulative-performance and drawdown comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
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


def _drawdown_from_wealth(wealth: pd.Series) -> pd.Series:
    values = wealth.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
    return pd.Series(values / peaks - 1.0, index=wealth.index, name="drawdown")


def plot_cumulative_performance(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot the required three cumulative-return curves plus their drawdown histories."""
    _validate_comparison(comparison)
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    cumulative_axis, drawdown_axis = axes
    try:
        for column in COMPARISON_COLUMNS:
            wealth = cumulative_wealth(comparison[column])
            label = DISPLAY_LABELS[column]
            cumulative_axis.plot(comparison.index, wealth - 1.0, label=label, linewidth=1.35)
            drawdown_axis.plot(
                comparison.index,
                _drawdown_from_wealth(wealth),
                label=label,
                linewidth=1.0,
            )

        cumulative_axis.axhline(0.0, linewidth=0.8)
        cumulative_axis.set_title("Step 5 Performance Comparison")
        cumulative_axis.set_ylabel("Cumulative return")
        cumulative_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cumulative_axis.legend(ncol=3)
        cumulative_axis.grid(True, alpha=0.22)

        drawdown_axis.axhline(0.0, linewidth=0.8)
        drawdown_axis.set_xlabel("Date")
        drawdown_axis.set_ylabel("Drawdown")
        drawdown_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        drawdown_axis.grid(True, alpha=0.22)

        locator = mdates.AutoDateLocator(  # type: ignore[no-untyped-call]
            minticks=5,
            maxticks=9,
        )
        drawdown_axis.xaxis.set_major_locator(locator)
        formatter = mdates.ConciseDateFormatter(locator)  # type: ignore[no-untyped-call]
        drawdown_axis.xaxis.set_major_formatter(formatter)
        figure.tight_layout()
        figure.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(figure)
