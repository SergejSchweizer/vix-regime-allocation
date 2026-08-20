"""Canonical predictive comparison and one-step regime-probability figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vix_regime_allocation.benchmarks import build_equal_weight_monthly_returns
from vix_regime_allocation.performance import cumulative_wealth

from .artifacts import load_step1_data, write_predictive_manifest
from .config import ASSET_ORDER
from .returns import asset_simple_returns

CUMULATIVE_FIGURE = "reports/predictive/figures/cumulative_performance_all_instruments.png"
PROBABILITY_FIGURE = "reports/predictive/figures/regime_forecast_probabilities.png"


def _daily_index(daily: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(daily, pd.DataFrame) or len(daily) < 2:
        raise ValueError("selected_test_daily must contain at least two rows.")
    required = {"return_date", "gross_return", "net_return"}
    if not required.issubset(daily.columns):
        raise ValueError("selected_test_daily is missing required return columns.")
    index = pd.DatetimeIndex(pd.to_datetime(daily["return_date"]), name="Date")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("selected_test_daily return dates must be unique and sorted.")
    return index


def comparison_return_frame(data: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Build the identical-date six-series final-test return comparison."""

    index = _daily_index(daily)
    assets = asset_simple_returns(data).loc[index, list(ASSET_ORDER)]
    equal = build_equal_weight_monthly_returns(data, index)
    result = pd.DataFrame(
        {
            "Predictive gross": daily["gross_return"].to_numpy(dtype=float),
            "Predictive net": daily["net_return"].to_numpy(dtype=float),
            "TLT": assets["TLT"].to_numpy(dtype=float),
            "GLD": assets["GLD"].to_numpy(dtype=float),
            "SPY": assets["SPY"].to_numpy(dtype=float),
            "Equal weight": equal.to_numpy(dtype=float),
        },
        index=index,
    )
    values = result.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("comparison returns must be finite and greater than -1.")
    return result


def _wealth_frame(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {column: cumulative_wealth(returns[column]).to_numpy(dtype=float) for column in returns},
        index=returns.index,
    )


def _drawdown_frame(wealth: pd.DataFrame) -> pd.DataFrame:
    result: dict[str, np.ndarray] = {}
    for column in wealth:
        values = wealth[column].to_numpy(dtype=float)
        peaks = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
        result[column] = values / peaks - 1.0
    return pd.DataFrame(result, index=wealth.index)


def plot_cumulative_performance(
    data: pd.DataFrame, daily: pd.DataFrame, output_path: Path
) -> Path:
    """Plot cumulative wealth, drawdown, and terminal return for every strategy/benchmark."""

    returns = comparison_return_frame(data, daily)
    wealth = _wealth_frame(returns)
    drawdown = _drawdown_frame(wealth)
    terminal = (wealth.iloc[-1] - 1.0) * 100.0

    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.0])
    cumulative_axis = figure.add_subplot(grid[0, 0])
    drawdown_axis = figure.add_subplot(grid[1, 0], sharex=cumulative_axis)
    terminal_axis = figure.add_subplot(grid[2, 0])

    for column in wealth:
        cumulative_axis.plot(wealth.index, wealth[column], label=column, linewidth=1.5)
        drawdown_axis.plot(drawdown.index, drawdown[column] * 100.0, label=column, linewidth=1.0)
    cumulative_axis.set_title("Predictive Holdout — Cumulative Performance Comparison")
    cumulative_axis.set_ylabel("Wealth (start = 1.0)")
    cumulative_axis.grid(alpha=0.25)
    cumulative_axis.legend(ncol=3, fontsize=8)

    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.grid(alpha=0.25)

    positions = np.arange(len(terminal))
    terminal_axis.barh(positions, terminal.to_numpy(dtype=float))
    terminal_axis.set_yticks(positions, terminal.index.tolist())
    terminal_axis.set_xlabel("Terminal cumulative return (%)")
    terminal_axis.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, terminal.to_numpy(dtype=float), strict=True):
        offset = 3.0 if value >= 0.0 else -3.0
        alignment = "left" if value >= 0.0 else "right"
        terminal_axis.text(value + offset, position, f"{value:.1f}%", va="center", ha=alignment)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("predictive cumulative comparison figure was not written.")
    return output


def probability_columns(daily: pd.DataFrame) -> list[str]:
    """Return the exact contiguous p_state_0..p_state_K columns."""

    columns = [column for column in daily.columns if column.startswith("p_state_")]
    columns = sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))
    expected = [f"p_state_{state}" for state in range(len(columns))]
    if columns != expected or len(columns) not in (2, 3):
        raise ValueError("daily artifact must contain contiguous K=2 or K=3 state probabilities.")
    return columns


def plot_regime_forecast_probabilities(daily: pd.DataFrame, output_path: Path) -> Path:
    """Plot the causal one-step-ahead regime probabilities used for final-test decisions."""

    if "decision_date" not in daily.columns:
        raise ValueError("daily artifact must contain decision_date.")
    columns = probability_columns(daily)
    index = pd.DatetimeIndex(pd.to_datetime(daily["decision_date"]), name="Date")
    probabilities = daily.loc[:, columns].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("forecast probabilities must be finite and normalized.")

    figure, axis = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    for position, column in enumerate(columns):
        axis.plot(index, probabilities[:, position], label=column, linewidth=1.3)
    axis.set_title("Selected Predictive Model — One-Step Regime Forecast Probabilities")
    axis.set_ylabel("Probability")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=len(columns))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("predictive regime-probability figure was not written.")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data = load_step1_data(root / "data/processed/step1_data.csv")
    daily = pd.read_csv(root / "reports/predictive/tables/selected_test_daily.csv")
    plot_cumulative_performance(data, daily, root / CUMULATIVE_FIGURE)
    plot_regime_forecast_probabilities(daily, root / PROBABILITY_FIGURE)
    write_predictive_manifest(root, figure_paths=(CUMULATIVE_FIGURE, PROBABILITY_FIGURE))
    print("Predictive comparison and regime-probability figures written.")


if __name__ == "__main__":
    main()
