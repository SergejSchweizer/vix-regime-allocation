"""Presentation-only helpers for the canonical HMM GWP2 notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


def _table_caption(number: int, title: str, description: str) -> str:
    """Return a paper-style caption with a stable table number."""
    return (
        '<p style="margin: 1em 0 0.25em; font-size: 0.9em; line-height: 1.35;">'
        f"<strong>Table {number}. {title}.</strong> {description}</p>"
    )


def _display_table(
    frame: pd.DataFrame,
    *,
    number: int,
    title: str,
    description: str,
    percent_columns: tuple[str, ...] = (),
    decimal_columns: tuple[str, ...] = (),
) -> None:
    """Display a numbered, described table with publication-ready formatting."""
    from IPython.display import HTML, display

    display(HTML(_table_caption(number, title, description)))
    formatters: dict[str, object] = {
        column: "{:.2%}" for column in percent_columns if column in frame.columns
    }
    formatters.update({column: "{:.2f}" for column in decimal_columns if column in frame.columns})
    styled = frame.style.format(formatters, na_rep="—").set_table_styles(
        [
            {
                "selector": "",
                "props": [("width", "100%"), ("table-layout", "fixed")],
            },
            {
                "selector": "th, td",
                "props": [
                    ("font-size", "8px"),
                    ("line-height", "1.2"),
                    ("overflow-wrap", "anywhere"),
                    ("padding", "3px"),
                ],
            },
        ]
    )
    display(HTML(styled.to_html()))


def _root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def _indexed_csv(relative: str) -> pd.DataFrame:
    frame = pd.read_csv(_root() / relative, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def step_1_data_overview() -> None:  # pragma: no cover - notebook presentation
    """Display the committed Step 1 sample and descriptive statistics."""
    from IPython.display import Markdown, display

    data = _indexed_csv("data/processed/step1_data.csv")
    display(Markdown("### Step 1 — Common sample and transformed series"))
    _display_table(
        data.head(),
        number=1,
        title="Canonical input sample preview",
        description=(
            "The first five observations of the aligned ETF and VIX dataset used throughout "
            "the analysis."
        ),
        percent_columns=("TLT_log_return", "GLD_log_return", "SPY_log_return"),
        decimal_columns=("VIX", "VIX_change"),
    )
    columns = ["TLT_log_return", "GLD_log_return", "SPY_log_return", "VIX_change"]
    summary = data[columns].describe().T.reset_index(names="series").astype("object")
    statistic_columns = ("mean", "std", "min", "25%", "50%", "75%", "max")
    for row, series in summary["series"].items():
        summary.at[row, "count"] = f"{summary.at[row, 'count']:,.0f}"
        for column in statistic_columns:
            value = float(summary.at[row, column])
            summary.at[row, column] = f"{value:.2f}" if series == "VIX_change" else f"{value:.2%}"
    _display_table(
        summary,
        number=2,
        title="Descriptive statistics for transformed series",
        description="Summary statistics for daily ETF log returns and daily changes in the VIX.",
    )


def step_2_hmm_diagnostics() -> None:  # pragma: no cover
    """Display persisted HMM K=2/K=3 parameters, transitions, states, and figures."""
    from IPython.display import Image, display

    root = _root()
    for n_states in (2, 3):
        parameters = pd.read_csv(root / f"reports/tables/step2_hmm_{n_states}_parameters.csv")
        transition = pd.read_csv(root / f"reports/tables/step2_hmm_{n_states}_transition.csv")
        _display_table(
            parameters,
            number=3 if n_states == 2 else 5,
            title=f"Gaussian HMM parameter estimates ({n_states} states)",
            description=(
                "Estimated state-specific VIX-change moments, initial probabilities, and "
                "decoded-state occupancy."
            ),
            percent_columns=(
                "start_probability",
                "viterbi_occupancy",
                "posterior_mean_probability",
            ),
            decimal_columns=("mean_vix_change", "variance_vix_change"),
        )
        _display_table(
            transition,
            number=4 if n_states == 2 else 6,
            title=f"Gaussian HMM transition matrix ({n_states} states)",
            description="Estimated one-day transition probabilities between decoded HMM states.",
            percent_columns=tuple(
                column for column in transition.columns if column != "from_state"
            ),
        )

    display(Image(filename=str(root / "reports/figures/step2_hmm_vix_states.png")))
    display(Image(filename=str(root / "reports/figures/step2_hmm_smoothed_probabilities.png")))


def step_3_hmm_selection() -> None:  # pragma: no cover
    """Display the HMM-only model comparison, selected state path, and ETF statistics."""
    from IPython.display import Image, Markdown, display

    root = _root()
    comparison = pd.read_csv(root / "reports/tables/step3_model_comparison.csv")
    selected = cast(
        dict[str, object],
        json.loads(
            (root / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
        ),
    )
    if selected.get("family") != "hmm":
        raise RuntimeError("Canonical selected model must be HMM.")
    states = _indexed_csv("reports/tables/step3_selected_states.csv")
    statistics = pd.read_csv(root / "reports/tables/step3_state_asset_statistics.csv")

    display(Markdown("### Step 3 — HMM-only model selection"))
    _display_table(
        comparison,
        number=7,
        title="HMM candidate comparison",
        description=(
            "Information criteria and diagnostic validity for the two candidate state counts."
        ),
        percent_columns=("min_viterbi_occupancy",),
        decimal_columns=("log_likelihood", "aic", "bic"),
    )
    _display_table(
        pd.Series(selected, name="value").rename_axis("selection_field").reset_index(),
        number=8,
        title="Selected HMM specification",
        description="The final state-count decision and the canonical artifact used downstream.",
    )
    _display_table(
        states.head().reset_index(),
        number=9,
        title="Decoded-state path preview",
        description=(
            "The first five dates of the Viterbi-decoded state series for the selected HMM."
        ),
    )
    _display_table(
        statistics,
        number=10,
        title="State-conditional ETF return statistics",
        description=(
            "Mean and standard deviation of daily ETF log returns conditional on the selected "
            "HMM state."
        ),
        percent_columns=("mean_log_return", "std_log_return"),
    )
    display(Image(filename=str(root / "reports/figures/step3_state_asset_statistics.png")))


def step_4_dual_allocations() -> None:  # pragma: no cover
    """Display the exact 100% Keep and 60/40 Spread allocation mappings."""
    root = _root()
    allocations = {
        "100_keep": pd.read_csv(root / "reports/tables/step4_allocation_100_keep.csv"),
        "60_40_spread": pd.read_csv(root / "reports/tables/step4_allocation_60_40_spread.csv"),
    }
    _display_table(
        allocations["100_keep"],
        number=11,
        title="100% Keep allocation by HMM state",
        description="Each state is allocated entirely to its highest-ranked ETF.",
        percent_columns=(
            "rank_1_mean_log_return",
            "rank_2_mean_log_return",
            "TLT_weight",
            "GLD_weight",
            "SPY_weight",
        ),
    )
    _display_table(
        allocations["60_40_spread"],
        number=12,
        title="60/40 Spread allocation by HMM state",
        description="Each state allocates 60% to the top-ranked ETF and 40% to the runner-up.",
        percent_columns=(
            "rank_1_mean_log_return",
            "rank_2_mean_log_return",
            "TLT_weight",
            "GLD_weight",
            "SPY_weight",
        ),
    )


def step_5_hmm_dual_method_comparison() -> None:  # pragma: no cover
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
    _display_table(
        summary,
        number=13,
        title="Backtest performance of HMM allocations and benchmarks",
        description=(
            "Cumulative and annualized performance over the common one-observation-lag "
            "evaluation sample."
        ),
        percent_columns=(
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
        ),
        decimal_columns=("sharpe_ratio",),
    )
    display(Image(filename=str(root / "reports/figures/step5_cumulative_performance.png")))


def canonical_works_cited() -> Any:  # pragma: no cover
    """Expose the bibliography registry path for the notebook's final Works Cited section."""
    from IPython.display import Markdown, display

    path = _root() / "reports/references.bib"
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError("reports/references.bib is missing or empty.")
    return display(Markdown(f"Bibliography registry: `{path.relative_to(_root()).as_posix()}`"))
