"""Step 3 state-conditional ETF return figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .model_config import SUPPORTED_STATE_COUNTS
from .state_statistics import ASSET_ORDER, STATISTICS_COLUMNS

BASIS_POINTS = 10_000.0


def _validate_statistics(statistics: pd.DataFrame) -> int:
    if not isinstance(statistics, pd.DataFrame):
        raise TypeError("statistics must be a pandas DataFrame.")
    if tuple(statistics.columns) != STATISTICS_COLUMNS:
        raise ValueError("statistics columns do not match the canonical Step 3 schema.")
    if len(statistics) == 0:
        raise ValueError("statistics must contain observations.")
    if not is_integer_dtype(statistics["state"].dtype):
        raise ValueError("statistics state column must use an integer dtype.")
    if not is_integer_dtype(statistics["observations"].dtype):
        raise ValueError("statistics observations column must use an integer dtype.")
    for column in ("mean_log_return", "std_log_return"):
        if not is_numeric_dtype(statistics[column].dtype):
            raise ValueError(f"statistics column {column!r} must be numeric.")
    states = np.sort(statistics["state"].unique().astype(int))
    if len(states) not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"statistics must contain exactly one of {SUPPORTED_STATE_COUNTS} states.")
    n_states = int(len(states))
    if not np.array_equal(states, np.arange(n_states, dtype=int)):
        raise ValueError("statistics states must be contiguous labels starting at zero.")
    expected = [(state, asset) for state in range(n_states) for asset in ASSET_ORDER]
    actual = list(zip(statistics["state"], statistics["asset"], strict=True))
    if actual != expected:
        raise ValueError(
            "statistics rows must use fixed state-major TLT/GLD/SPY order exactly once."
        )
    values = statistics[["mean_log_return", "std_log_return"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("statistics mean and standard-deviation values must be finite.")
    if np.any(statistics["std_log_return"].to_numpy(dtype=float) < 0.0):
        raise ValueError("statistics standard deviations cannot be negative.")
    if np.any(statistics["observations"].to_numpy(dtype=int) <= 0):
        raise ValueError("statistics observation counts must be positive.")
    return n_states


def plot_state_asset_statistics(
    statistics: pd.DataFrame, output_path: Path, *, model_label: str | None = None
) -> None:
    """Plot conditional means and dispersion separately, both in daily basis points.

    Sample standard deviation is intentionally not drawn as an error bar around the mean:
    it is dispersion of individual daily returns, not standard error or a confidence interval.
    """
    n_states = _validate_statistics(statistics)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_positions = np.arange(n_states, dtype=float)
    width = 0.24
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    mean_axis, std_axis = axes
    try:
        for asset_index, asset in enumerate(ASSET_ORDER):
            rows = statistics.loc[statistics["asset"] == asset]
            offset = (asset_index - 1) * width
            positions = state_positions + offset
            mean_axis.bar(
                positions,
                rows["mean_log_return"].to_numpy(dtype=float) * BASIS_POINTS,
                width=width,
                label=asset,
            )
            std_axis.bar(
                positions,
                rows["std_log_return"].to_numpy(dtype=float) * BASIS_POINTS,
                width=width,
                label=asset,
            )

        mean_axis.axhline(0.0, linewidth=0.9)
        mean_axis.set_ylabel("Mean daily log return (bp)")
        title = (
            f"{model_label}: state-conditional ETF return means and dispersion"
            if model_label
            else "State-conditional ETF return means and dispersion"
        )
        mean_axis.set_title(title)
        mean_axis.grid(True, axis="y", alpha=0.22)
        mean_axis.legend(title="ETF", ncol=3)

        std_axis.set_xticks(
            state_positions,
            [f"State {state}" for state in range(n_states)],
        )
        std_axis.set_xlabel("HMM state" if model_label else "Preferred-model state")
        std_axis.set_ylabel("Sample daily standard deviation (bp)")
        std_axis.grid(True, axis="y", alpha=0.22)
        std_axis.legend(title="ETF", ncol=3)

        figure.tight_layout()
        figure.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(figure)
