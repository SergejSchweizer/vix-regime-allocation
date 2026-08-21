"""Presentation-only helpers for the canonical HMM GWP2 notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def _indexed_csv(relative: str) -> pd.DataFrame:
    frame = pd.read_csv(_root() / relative, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def step_1_data_overview() -> Any:  # pragma: no cover - notebook presentation
    """Display the committed Step 1 sample and descriptive statistics."""
    from IPython.display import Markdown, display

    data = _indexed_csv("data/processed/step1_data.csv")
    display(Markdown("### Step 1 — Common sample and transformed series"))
    display(data.head())
    columns = ["TLT_log_return", "GLD_log_return", "SPY_log_return", "VIX_change"]
    display(data[columns].describe().T)
    return data


def step_2_hmm_diagnostics() -> dict[int, dict[str, pd.DataFrame]]:  # pragma: no cover
    """Display persisted HMM K=2/K=3 parameters, transitions, states, and figures."""
    from IPython.display import Image, Markdown, display

    root = _root()
    diagnostics: dict[int, dict[str, pd.DataFrame]] = {}
    for n_states in (2, 3):
        parameters = pd.read_csv(root / f"reports/tables/step2_hmm_{n_states}_parameters.csv")
        transition = pd.read_csv(root / f"reports/tables/step2_hmm_{n_states}_transition.csv")
        states = _indexed_csv(f"reports/tables/step2_hmm_{n_states}_states.csv")
        diagnostics[n_states] = {
            "parameters": parameters,
            "transition": transition,
            "states": states,
        }
        display(Markdown(f"### Gaussian HMM candidate: K={n_states}"))
        display(parameters)
        display(transition)

    display(Image(filename=str(root / "reports/figures/step2_hmm_vix_states.png")))
    display(Image(filename=str(root / "reports/figures/step2_hmm_smoothed_probabilities.png")))
    return diagnostics


def step_3_hmm_selection() -> dict[str, object]:  # pragma: no cover
    """Display the HMM-only model comparison, selected state path, and ETF statistics."""
    from IPython.display import Image, Markdown, display

    root = _root()
    comparison = pd.read_csv(root / "reports/tables/step3_model_comparison.csv")
    selected = json.loads(
        (root / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
    )
    if selected.get("family") != "hmm":
        raise RuntimeError("Canonical selected model must be HMM.")
    states = _indexed_csv("reports/tables/step3_selected_states.csv")
    statistics = pd.read_csv(root / "reports/tables/step3_state_asset_statistics.csv")

    display(Markdown("### Step 3 — HMM-only model selection"))
    display(comparison)
    display(pd.Series(selected, name="selected_model").to_frame())
    display(states.head())
    display(statistics)
    display(Image(filename=str(root / "reports/figures/step3_state_asset_statistics.png")))
    return selected


def step_4_dual_allocations() -> dict[str, pd.DataFrame]:  # pragma: no cover
    """Display the exact 100% Keep and 60/40 Spread allocation mappings."""
    from IPython.display import Markdown, display

    root = _root()
    allocations = {
        "100_keep": pd.read_csv(root / "reports/tables/step4_allocation_100_keep.csv"),
        "60_40_spread": pd.read_csv(root / "reports/tables/step4_allocation_60_40_spread.csv"),
    }
    display(Markdown("### Step 4 — 100% Keep"))
    display(allocations["100_keep"])
    display(Markdown("### Step 4 — 60/40 Spread"))
    display(allocations["60_40_spread"])
    return allocations


def step_5_hmm_dual_method_comparison() -> dict[str, pd.DataFrame]:  # pragma: no cover
    """Display four aligned Step 5 portfolios, metrics, and cumulative performance."""
    from IPython.display import Image, Markdown, display

    root = _root()
    daily = _indexed_csv("reports/tables/step5_daily_returns.csv")
    summary = pd.read_csv(root / "reports/tables/step5_performance_summary.csv")
    expected = [
        "hmm_100_keep",
        "hmm_60_40_spread",
        "equal_weight_monthly",
        "spy_buy_hold",
    ]
    if daily.columns.tolist() != expected:
        raise RuntimeError("Canonical Step 5 daily returns do not have the four-portfolio schema.")
    display(Markdown("### Step 5 — One-observation-lag portfolio comparison"))
    display(summary)
    display(Image(filename=str(root / "reports/figures/step5_cumulative_performance.png")))
    return {"daily": daily, "summary": summary}


def canonical_works_cited() -> Any:  # pragma: no cover
    """Expose the bibliography registry path for the notebook's final Works Cited section."""
    from IPython.display import Markdown, display

    path = _root() / "reports/references.bib"
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError("reports/references.bib is missing or empty.")
    return display(Markdown(f"Bibliography registry: `{path.relative_to(_root()).as_posix()}`"))
