"""Presentation-only helpers for the canonical MScFE technical notebook.

The notebook is intentionally thin: it contains explanatory Markdown and one-line
calls into this module.  All file loading, validation, formatting, and display
logic lives here so the notebook itself never contains analysis implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from .transform import OUTPUT_COLUMNS


_REQUIRED_STEP5_COLUMNS: tuple[str, ...] = (
    "portfolio",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from the current working directory or ``start``."""
    candidate = (start or Path.cwd()).resolve()
    for root in (candidate, *candidate.parents):
        if (root / "pyproject.toml").is_file() and (root / "reports").is_dir():
            return root
    raise FileNotFoundError("Could not locate the vix-regime-allocation repository root.")


def _read_csv(relative_path: str, *, index_date: bool = False) -> pd.DataFrame:
    path = find_repo_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical artifact: {relative_path}")
    if index_date:
        frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        frame.index = pd.DatetimeIndex(frame.index, name="Date")
        return frame
    return pd.read_csv(path)


def _read_json(relative_path: str) -> dict[str, Any]:
    path = find_repo_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical artifact: {relative_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {relative_path}.")
    return payload


def _show_image(relative_path: str, *, width: int = 1050) -> None:
    path = find_repo_root() / relative_path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty canonical figure: {relative_path}")
    display(Image(filename=str(path), width=width))


def _show_heading(text: str) -> None:
    display(Markdown(f"**{text}**"))


def show_step1_sample() -> None:
    """Display the validated common sample and a compact sample summary."""
    data = _read_csv("data/processed/step1_data.csv", index_date=True)
    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("Step 1 data does not match the canonical schema.")
    values = data.to_numpy(dtype=float)
    if data.empty or np.any(~np.isfinite(values)):
        raise ValueError("Step 1 data must be non-empty and finite.")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Step 1 Date index must be unique and sorted ascending.")

    summary = pd.DataFrame(
        {
            "value": [
                data.index.min().date().isoformat(),
                data.index.max().date().isoformat(),
                len(data),
                int(data.isna().sum().sum()),
            ]
        },
        index=["start_date", "end_date", "observations", "missing_values"],
    )
    _show_heading("First five observations of the canonical common sample")
    display(data.head())
    _show_heading("Sample integrity summary")
    display(summary)


def show_step1_figures() -> None:
    """Display the canonical Step 1 exploratory figures."""
    _show_heading("ETF daily log returns")
    _show_image("reports/figures/step1_etf_log_returns.png")
    _show_heading("Daily VIX first difference")
    _show_image("reports/figures/step1_vix_change.png")


def show_step2_markov_results() -> None:
    """Display the persisted K=2 and K=3 quantile-Markov outputs."""
    for n_states in (2, 3):
        _show_heading(f"Quantile Markov specification, K={n_states}")
        display(_read_csv(f"reports/tables/step2_markov_{n_states}_thresholds.csv"))
        display(_read_csv(f"reports/tables/step2_markov_{n_states}_transition.csv"))
        display(_read_csv(f"reports/tables/step2_markov_{n_states}_stationary.csv"))
    _show_image("reports/figures/step2_markov_vix_states.png")


def show_step2_hmm_results() -> None:
    """Display the persisted K=2 and K=3 Gaussian-HMM outputs."""
    for n_states in (2, 3):
        _show_heading(f"Gaussian HMM specification, K={n_states}")
        display(_read_csv(f"reports/tables/step2_hmm_{n_states}_parameters.csv"))
        display(_read_csv(f"reports/tables/step2_hmm_{n_states}_transition.csv"))
    _show_image("reports/figures/step2_hmm_vix_states.png")
    _show_image("reports/figures/step2_hmm_smoothed_probabilities.png")


def show_step3_model_selection() -> None:
    """Display candidate comparison and the deterministic selected-model provenance."""
    comparison = _read_csv("reports/tables/step3_model_comparison.csv")
    selected = _read_json("reports/generated/step3_selected_model.json")
    _show_heading("Candidate comparison; information criteria are interpreted within family")
    display(comparison)
    _show_heading("Selected-model provenance")
    display(pd.DataFrame([selected]))


def show_step3_state_statistics() -> None:
    """Display preferred-state ETF moments and their canonical figure."""
    statistics = _read_csv("reports/tables/step3_state_asset_statistics.csv")
    _show_heading("State-conditional ETF daily log-return statistics")
    display(statistics)
    _show_image("reports/figures/step3_state_asset_statistics.png")


def show_step4_allocation() -> None:
    """Display the persisted state-to-ETF allocation mapping."""
    allocation = _read_csv("reports/tables/step4_allocation_mapping.csv")
    _show_heading("Deterministic state-to-allocation mapping")
    display(allocation)


def show_step5_backtest() -> None:
    """Display canonical backtest metrics and cumulative-performance figure."""
    summary = _read_csv("reports/tables/step5_performance_summary.csv")
    if tuple(summary.columns) != _REQUIRED_STEP5_COLUMNS:
        raise ValueError("Step 5 performance summary does not match the canonical schema.")
    pretty = summary.copy()
    for column in (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    ):
        pretty[column] = pretty[column].map(lambda value: f"{100.0 * float(value):.2f}%")
    pretty["sharpe_ratio"] = pretty["sharpe_ratio"].map(lambda value: f"{float(value):.3f}")
    _show_heading("Step 5 performance summary")
    display(pretty)
    _show_image("reports/figures/step5_cumulative_performance.png")


def show_step5_sensitivity() -> None:
    """Display the state-count sensitivity table when the canonical artifact exists."""
    path = find_repo_root() / "reports/tables/step5_state_count_sensitivity.csv"
    if not path.is_file():
        display(
            Markdown(
                "*The canonical K=2 versus K=3 Step 5 sensitivity artifact has not yet "
                "been generated on this branch.*"
            )
        )
        return
    display(pd.read_csv(path))


__all__ = [
    "find_repo_root",
    "show_step1_sample",
    "show_step1_figures",
    "show_step2_markov_results",
    "show_step2_hmm_results",
    "show_step3_model_selection",
    "show_step3_state_statistics",
    "show_step4_allocation",
    "show_step5_backtest",
    "show_step5_sensitivity",
]
