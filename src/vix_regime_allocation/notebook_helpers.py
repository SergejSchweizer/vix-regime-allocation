"""Presentation-only helpers for the canonical MScFE technical notebook.

The notebook is intentionally thin: it contains explanatory Markdown and one-line
calls into this module. All file loading, validation, formatting, interpretation,
and display logic lives here so the notebook itself never contains analysis implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from .model_config import HMM_MIN_STATE_OCCUPANCY
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


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


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
    display(
        Markdown(
            f"The common sample contains **{len(data):,} daily observations** from "
            f"**{data.index.min().date().isoformat()}** through "
            f"**{data.index.max().date().isoformat()}**, with no missing values after "
            "the common-date filter and lag construction."
        )
    )


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

    reason = selected.get("selection_reason")
    if isinstance(reason, str) and reason:
        display(Markdown(f"**Decision-rule outcome.** {reason}"))

    hmm_best = selected.get("hmm_best_n_states")
    if hmm_best in (2, 3):
        parameters = _read_csv(f"reports/tables/step2_hmm_{int(hmm_best)}_parameters.csv")
        if "viterbi_occupancy" in parameters.columns:
            minimum = float(parameters["viterbi_occupancy"].min())
            gap = HMM_MIN_STATE_OCCUPANCY - minimum
            if gap > 0.0:
                display(
                    Markdown(
                        "The HMM fallback is a **project diagnostic rule**, not proof that the "
                        f"candidate is statistically impossible. Its smallest decoded state has "
                        f"occupancy **{100.0 * minimum:.4f}%**, which is only "
                        f"**{100.0 * gap:.4f} percentage points** below the fixed "
                        f"{100.0 * HMM_MIN_STATE_OCCUPANCY:.0f}% threshold."
                    )
                )


def show_step3_state_statistics() -> None:
    """Display preferred-state ETF moments, leaders, and the canonical figure."""
    statistics = _read_csv("reports/tables/step3_state_asset_statistics.csv")
    _show_heading("State-conditional ETF daily log-return statistics")
    display(statistics)

    leader_lines: list[str] = []
    for state, rows in statistics.groupby("state", sort=True):
        leader = rows.loc[rows["mean_log_return"].astype(float).idxmax()]
        mean_bps = 10_000.0 * float(leader["mean_log_return"])
        leader_lines.append(
            f"- State {int(state)}: **{leader['asset']}** has the largest historical "
            f"conditional mean, {mean_bps:.3f} basis points per day."
        )
    display(Markdown("**Conditional-mean leaders**\n\n" + "\n".join(leader_lines)))
    _show_image("reports/figures/step3_state_asset_statistics.png")


def show_step4_allocation() -> None:
    """Display and interpret the persisted state-to-ETF allocation mapping."""
    allocation = _read_csv("reports/tables/step4_allocation_mapping.csv")
    _show_heading("Deterministic state-to-allocation mapping")
    display(allocation)
    mapping = ", ".join(
        f"State {int(row.state)} → {row.selected_asset}"
        for row in allocation.itertuples(index=False)
    )
    display(
        Markdown(
            f"The implemented mapping is **{mapping}**. Because each row is one-hot, the "
            "strategy is fully concentrated in a single ETF at every decision date."
        )
    )


def show_step5_backtest() -> None:
    """Display canonical backtest metrics, interpretation, and cumulative figure."""
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
        pretty[column] = pretty[column].map(lambda value: _percent(float(value)))
    pretty["sharpe_ratio"] = pretty["sharpe_ratio"].map(lambda value: f"{float(value):.3f}")
    _show_heading("Step 5 performance summary")
    display(pretty)

    indexed = summary.set_index("portfolio")
    required = {"regime_rotation", "equal_weight_monthly", "spy_buy_hold"}
    if required.issubset(set(indexed.index.astype(str))):
        rotation = indexed.loc["regime_rotation"]
        equal_weight = indexed.loc["equal_weight_monthly"]
        spy = indexed.loc["spy_buy_hold"]
        display(
            Markdown(
                "**Economic interpretation.** The lagged regime rotation is not competitive "
                "with either required benchmark on this historical sample. It earns "
                f"**{_percent(float(rotation['annualized_return']))} annualized** with Sharpe "
                f"**{float(rotation['sharpe_ratio']):.3f}**, versus "
                f"**{_percent(float(equal_weight['annualized_return']))} / "
                f"{float(equal_weight['sharpe_ratio']):.3f}** for monthly equal weight and "
                f"**{_percent(float(spy['annualized_return']))} / "
                f"{float(spy['sharpe_ratio']):.3f}** for SPY. The equal-weight benchmark also "
                f"has the smallest maximum drawdown at **{_percent(float(equal_weight['max_drawdown']))}**. "
                "This is consistent with the distinction between a strong same-day state/return "
                "association and genuinely useful next-observation predictive information."
            )
        )
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
    sensitivity = pd.read_csv(path)
    display(sensitivity)
    if {"n_states", "sharpe_ratio"}.issubset(sensitivity.columns):
        best = sensitivity.loc[sensitivity["sharpe_ratio"].astype(float).idxmax()]
        display(
            Markdown(
                f"The highest historical Sharpe ratio in this sensitivity table occurs at "
                f"**K={int(best['n_states'])}**, but this is a robustness comparison only; it "
                "does not replace the pre-declared Step 3 state-count selection rule."
            )
        )


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
