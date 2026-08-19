"""Grouped Step 3 state-conditional ETF return figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .model_config import SUPPORTED_STATE_COUNTS
from .state_statistics import ASSET_ORDER, STATISTICS_COLUMNS


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


def plot_state_asset_statistics(statistics: pd.DataFrame, output_path: Path) -> None:
    """Write grouped daily-mean bars with state/asset sample-standard-deviation error bars."""
    n_states = _validate_statistics(statistics)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_positions = np.arange(n_states, dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 6))
    for asset_index, asset in enumerate(ASSET_ORDER):
        rows = statistics.loc[statistics["asset"] == asset]
        offset = (asset_index - 1) * width
        axis.bar(
            state_positions + offset,
            rows["mean_log_return"].to_numpy(dtype=float),
            width=width,
            yerr=rows["std_log_return"].to_numpy(dtype=float),
            capsize=4,
            label=asset,
        )

    axis.axhline(0.0, linewidth=0.8)
    axis.set_xticks(state_positions, [f"State {state}" for state in range(n_states)])
    axis.set_xlabel("Preferred-model state")
    axis.set_ylabel("Daily ETF log return")
    axis.set_title(
        "State-conditional ETF mean daily log returns with sample-standard-deviation bars"
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(title="ETF")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
