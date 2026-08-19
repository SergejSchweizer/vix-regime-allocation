"""Canonical Step 1 exploratory figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_numeric_dtype

from vix_regime_allocation.transform import ETF_COLUMNS, OUTPUT_COLUMNS


def _validate_step1_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"data columns must be exactly {list(OUTPUT_COLUMNS)}.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a pandas DatetimeIndex.")
    if data.index.name != "Date":
        raise ValueError("data index must be named 'Date'.")
    if data.index.tz is not None:
        raise ValueError("data index must be timezone-naive.")
    if data.index.has_duplicates:
        raise ValueError("data index must not contain duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted in ascending order.")
    if data.empty:
        raise ValueError("data must contain at least one observation.")
    for column in OUTPUT_COLUMNS:
        if not is_numeric_dtype(data[column].dtype):
            raise ValueError(f"data column {column!r} must be numeric.")
    if np.any(~np.isfinite(data.to_numpy(dtype=float))):
        raise ValueError("Step 1 plotting data must contain only finite values.")


def _validate_output_path(output_path: Path) -> None:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must have a .png suffix.")


def _format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_etf_log_returns(data: pd.DataFrame, output_path: Path) -> None:
    """Save TLT/GLD/SPY daily log returns with percentage-scaled axes."""
    _validate_step1_data(data)
    _validate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    try:
        for asset in ETF_COLUMNS:
            ax.plot(data.index, data[f"{asset}_log_return"], label=asset, linewidth=0.75, alpha=0.8)
        ax.axhline(0.0, linewidth=0.9, linestyle="--")
        ax.set_title("ETF Daily Log Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily log return")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        _format_dates(ax)
        ax.legend(title="ETF", ncol=3)
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_vix_change(data: pd.DataFrame, output_path: Path) -> None:
    """Save the daily first difference of VIX in index points."""
    _validate_step1_data(data)
    _validate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    try:
        ax.plot(data.index, data["VIX_change"], label="Daily VIX change", linewidth=0.8)
        ax.axhline(0.0, linewidth=0.9, linestyle="--")
        ax.set_title("Daily Change in VIX")
        ax.set_xlabel("Date")
        ax.set_ylabel("Change in VIX (index points)")
        _format_dates(ax)
        ax.legend()
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)
