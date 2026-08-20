"""Step 5 cumulative-performance, drawdown, and terminal-outcome comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_numeric_dtype

from .performance import cumulative_wealth
from .state_statistics import ASSET_ORDER

COMPARISON_COLUMNS: tuple[str, ...] = (
    "regime_rotation",
    "equal_weight_monthly",
    "spy_buy_hold",
)
PLOT_COLUMNS: tuple[str, ...] = (
    "regime_rotation",
    "equal_weight_monthly",
    "TLT",
    "GLD",
    "SPY",
)
DISPLAY_LABELS: dict[str, str] = {
    "regime_rotation": "Regime rotation",
    "equal_weight_monthly": "Equal weight (monthly reset)",
    "TLT": "TLT buy and hold",
    "GLD": "GLD buy and hold",
    "SPY": "SPY buy and hold",
}
ENDPOINT_Y_OFFSETS: dict[str, int] = {
    "regime_rotation": 10,
    "equal_weight_monthly": -10,
    "TLT": 0,
    "GLD": 10,
    "SPY": -10,
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


def _validate_instrument_returns(
    instrument_returns: pd.DataFrame, comparison_index: pd.DatetimeIndex
) -> None:
    if not isinstance(instrument_returns, pd.DataFrame):
        raise TypeError("instrument_returns must be a pandas DataFrame.")
    if tuple(instrument_returns.columns) != ASSET_ORDER:
        raise ValueError("instrument_returns columns must be exactly TLT, GLD, SPY.")
    if not isinstance(instrument_returns.index, pd.DatetimeIndex):
        raise ValueError("instrument_returns index must be a pandas DatetimeIndex.")
    if not instrument_returns.index.equals(comparison_index):
        raise ValueError("instrument_returns must use exactly the comparison Date index.")
    for column in ASSET_ORDER:
        if not is_numeric_dtype(instrument_returns[column].dtype):
            raise ValueError(f"instrument return column {column!r} must be numeric.")
    values = instrument_returns.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("instrument simple returns must be finite and greater than -1.")


def _load_instrument_returns(
    comparison_index: pd.DatetimeIndex, output_path: Path
) -> pd.DataFrame:
    """Load TLT/GLD/SPY simple returns on the exact Step 5 comparison dates."""
    repo_root = output_path.resolve().parents[2]
    data_path = repo_root / "data/processed/step1_data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(
            "Step 1 data is required to plot all instruments: " f"{data_path}"
        )
    data = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")
    data.index = pd.DatetimeIndex(data.index, name="Date")
    log_columns = [f"{asset}_log_return" for asset in ASSET_ORDER]
    missing = [column for column in log_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Step 1 data is missing instrument return columns: {missing}")
    if not comparison_index.isin(data.index).all():
        raise ValueError("Step 1 data does not contain every Step 5 comparison date.")
    values = np.expm1(data.loc[comparison_index, log_columns].to_numpy(dtype=float))
    instrument_returns = pd.DataFrame(
        values,
        index=comparison_index,
        columns=list(ASSET_ORDER),
    )
    instrument_returns.index.name = "Date"
    _validate_instrument_returns(instrument_returns, comparison_index)
    return instrument_returns


def _drawdown_from_wealth(wealth: pd.Series) -> pd.Series:
    values = wealth.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
    return pd.Series(values / peaks - 1.0, index=wealth.index, name="drawdown")


def plot_cumulative_performance(
    comparison: pd.DataFrame,
    output_path: Path,
    instrument_returns: pd.DataFrame | None = None,
) -> None:
    """Plot strategy, benchmarks, and every investable instrument on common dates.

    The canonical Step 5 comparison remains the three required portfolio series. For
    the user-facing cumulative figure, TLT, GLD, and SPY buy-and-hold returns are shown
    together with regime rotation and the equal-weight benchmark. When instrument
    returns are not supplied explicitly, they are reconstructed from the canonical
    Step 1 data beside the report artifacts.
    """
    _validate_comparison(comparison)
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    comparison_index = pd.DatetimeIndex(comparison.index, name="Date")
    if instrument_returns is None:
        instrument_returns = _load_instrument_returns(comparison_index, output_path)
    else:
        _validate_instrument_returns(instrument_returns, comparison_index)
    if not np.allclose(
        comparison["spy_buy_hold"].to_numpy(dtype=float),
        instrument_returns["SPY"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("SPY instrument returns must match the canonical SPY benchmark.")

    plot_returns = pd.concat(
        [
            comparison[["regime_rotation", "equal_weight_monthly"]],
            instrument_returns.loc[:, list(ASSET_ORDER)],
        ],
        axis=1,
    ).loc[:, list(PLOT_COLUMNS)]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(2.2, 1.0),
        width_ratios=(2.2, 1.0),
    )
    cumulative_axis = figure.add_subplot(grid[0, :])
    drawdown_axis = figure.add_subplot(grid[1, 0], sharex=cumulative_axis)
    terminal_axis = figure.add_subplot(grid[1, 1])

    terminal_returns: dict[str, float] = {}
    try:
        for column in PLOT_COLUMNS:
            wealth = cumulative_wealth(plot_returns[column])
            label = DISPLAY_LABELS[column]
            cumulative_return = wealth - 1.0
            terminal_returns[column] = float(cumulative_return.iloc[-1])

            line = cumulative_axis.plot(
                comparison_index,
                cumulative_return,
                label=label,
                linewidth=1.45,
            )[0]
            drawdown_axis.plot(
                comparison_index,
                _drawdown_from_wealth(wealth),
                label=label,
                linewidth=1.0,
                color=line.get_color(),
            )
            cumulative_axis.annotate(
                f"{terminal_returns[column]:.1%}",
                xy=(comparison_index[-1], terminal_returns[column]),
                xytext=(7, ENDPOINT_Y_OFFSETS[column]),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=line.get_color(),
            )

        cumulative_axis.axhline(0.0, linewidth=0.8)
        cumulative_axis.set_title("Step 5 Cumulative Performance Comparison — All Instruments")
        cumulative_axis.set_ylabel("Cumulative return")
        cumulative_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cumulative_axis.legend(ncol=3)
        cumulative_axis.grid(True, alpha=0.22)
        cumulative_axis.tick_params(labelbottom=False)

        drawdown_axis.axhline(0.0, linewidth=0.8)
        drawdown_axis.set_xlabel("Date")
        drawdown_axis.set_ylabel("Drawdown")
        drawdown_axis.set_title("Drawdown history")
        drawdown_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        drawdown_axis.grid(True, alpha=0.22)

        locator = mdates.AutoDateLocator(  # type: ignore[no-untyped-call]
            minticks=3,
            maxticks=9,
        )
        drawdown_axis.xaxis.set_major_locator(locator)
        formatter = mdates.ConciseDateFormatter(locator)  # type: ignore[no-untyped-call]
        drawdown_axis.xaxis.set_major_formatter(formatter)

        terminal_values = np.array(
            [terminal_returns[column] for column in PLOT_COLUMNS],
            dtype=float,
        )
        terminal_labels = [DISPLAY_LABELS[column] for column in PLOT_COLUMNS]
        positions = np.arange(len(PLOT_COLUMNS), dtype=float)
        bars = terminal_axis.barh(positions, terminal_values)
        terminal_axis.set_yticks(positions, terminal_labels)
        terminal_axis.invert_yaxis()
        terminal_axis.set_title("Terminal cumulative return")
        terminal_axis.set_xlabel("Cumulative return")
        terminal_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        terminal_axis.axvline(0.0, linewidth=0.8)
        terminal_axis.grid(True, axis="x", alpha=0.22)
        for bar, value in zip(bars, terminal_values, strict=True):
            terminal_axis.text(
                value,
                bar.get_y() + bar.get_height() / 2.0,
                f" {value:.1%}",
                va="center",
                fontsize=8.5,
            )

        figure.savefig(output_path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(figure)
