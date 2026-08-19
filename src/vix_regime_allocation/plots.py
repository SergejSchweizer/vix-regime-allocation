"""Canonical Step 1 exploratory figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from vix_regime_allocation.transform import ETF_COLUMNS, OUTPUT_COLUMNS


def _validate_step1_data(data: pd.DataFrame) -> None:
    """Validate the exact transformer-to-plotter interface contract."""
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

    values = data.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Step 1 plotting data must contain only finite values.")


def _validate_output_path(output_path: Path) -> None:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must have a .png suffix.")


def plot_etf_log_returns(data: pd.DataFrame, output_path: Path) -> None:
    """Save the required TLT/GLD/SPY daily log-return figure."""
    _validate_step1_data(data)
    _validate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        for asset in ETF_COLUMNS:
            ax.plot(data.index, data[f"{asset}_log_return"], label=asset, linewidth=0.9)
        ax.axhline(0.0, linewidth=0.8, linestyle="--")
        ax.set_title("ETF Daily Log Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Log Return")
        ax.legend(title="ETF")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_vix_change(data: pd.DataFrame, output_path: Path) -> None:
    """Save the required daily VIX first-difference figure."""
    _validate_step1_data(data)
    _validate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        ax.plot(data.index, data["VIX_change"], label="VIX change", linewidth=0.9)
        ax.axhline(0.0, linewidth=0.8, linestyle="--")
        ax.set_title("Daily Change in VIX")
        ax.set_xlabel("Date")
        ax.set_ylabel("VIX Change")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
