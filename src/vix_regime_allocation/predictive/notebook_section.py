"""Presentation-only notebook section for the causal predictive extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

DisplayPayload = list[tuple[str, object]]


def predictive_conclusion(dominance: pd.DataFrame) -> str:
    """Render the empirical all-assets dominance conclusion without hard-coded success."""

    required = {"benchmark", "benchmark_cagr", "strategy_net_cagr", "cagr_difference"}
    if not isinstance(dominance, pd.DataFrame) or set(dominance.columns) != required:
        raise ValueError("dominance must contain the exact four-column comparison schema.")
    if set(dominance["benchmark"].astype(str)) != {"TLT", "GLD", "SPY"}:
        raise ValueError("dominance must compare exactly TLT, GLD, and SPY.")
    margin = float(dominance["cagr_difference"].min())
    if margin > 0.0:
        return (
            "On the untouched 2021+ holdout, the selected net predictive strategy beat "
            "TLT, GLD, and SPY in annualized return. The minimum CAGR advantage across "
            f"the three individual assets is {margin:.2%}. This is a historical holdout "
            "result under the fixed experiment and transaction-cost assumption; it is not "
            "a guarantee of future outperformance."
        )
    return (
        "On the untouched 2021+ holdout, the selected net predictive strategy did not beat "
        "every individual asset in annualized return. The minimum pairwise CAGR difference "
        f"is {margin:.2%}. Under the pre-registered protocol this result is not retuned; it "
        "means the VIX-change-only predictive specification did not establish all-asset "
        "outperformance on the final holdout."
    )


def _selected_text(selected: dict[str, object]) -> str:
    hurdle = cast(float, selected["switch_hurdle_bps"])
    cost = cast(float, selected["transaction_cost_bps"])
    return (
        f"**Selected validation configuration:** {selected['family']} K={selected['n_states']}, "
        f"switching hurdle {hurdle:.0f} bps, transaction cost {cost:.0f} bps one-way."
    )


def _display_payload(payload: DisplayPayload) -> None:
    from IPython.display import Image, Markdown, display

    for kind, value in payload:
        if kind == "markdown":
            display(Markdown(str(value)))
        elif kind == "image":
            display(Image(filename=str(value)))
        elif kind == "object":
            display(value)
        else:
            raise ValueError(f"Unknown display payload kind: {kind!r}.")


def _payload(
    selected: dict[str, object],
    validation: pd.DataFrame,
    performance: pd.DataFrame,
    dominance: pd.DataFrame,
    figure_dir: Path,
) -> DisplayPayload:
    introduction = (
        "## Predictive Extension — Causal One-Step Regime Allocation\n\n"
        "This section is an **additive research extension** to the required assignment "
        "analysis above. It does not replace or reinterpret Steps 1–5. The extension changes "
        "the trading question from contemporaneous state classification to a strictly "
        "chronological one-step forecast: information available through decision date *t* "
        "forecasts the regime distribution for the next observed row, which is then mapped "
        "to expected TLT, GLD, and SPY returns.\n\n"
        "The experiment is pre-registered in code: expanding-window training, monthly "
        "refits, validation from 2015 through 2020, a final untouched 2021+ holdout, four "
        "switching hurdles, fixed 5 bps one-way transaction costs, and no final-test "
        "retuning. Model family, state count, and switching hurdle are selected only by "
        "validation net mean log growth."
    )
    return [
        ("markdown", introduction),
        ("markdown", _selected_text(selected)),
        ("markdown", "### Validation candidate comparison"),
        ("object", validation),
        ("markdown", "### Final 2021+ holdout performance"),
        ("object", performance),
        ("markdown", "### CAGR comparison against every individual asset"),
        ("object", dominance),
        ("markdown", predictive_conclusion(dominance)),
        ("markdown", "### Holdout cumulative performance and drawdown"),
        ("image", figure_dir / "cumulative_performance_all_instruments.png"),
        ("markdown", "### One-step regime forecast probabilities used by the selected model"),
        ("image", figure_dir / "regime_forecast_probabilities.png"),
        (
            "markdown",
            "**Source note.** All predictive tables and figures are project calculations "
            "from the canonical Step 1 dataset. Every historical decision is independently "
            "audited to satisfy `training_end < decision_date < return_date`.",
        ),
    ]


def render_predictive_extension(root: Path | None = None) -> None:
    """Display canonical predictive artifacts without fitting, forecasting, or selecting."""

    repository_root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    table_dir = repository_root / "reports/predictive/tables"
    generated_dir = repository_root / "reports/predictive/generated"
    figure_dir = repository_root / "reports/predictive/figures"

    selected = json.loads((generated_dir / "selected_strategy.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(table_dir / "candidate_validation_summary.csv")
    performance = pd.read_csv(table_dir / "selected_test_performance.csv")
    dominance = pd.read_csv(table_dir / "test_asset_dominance.csv")
    _display_payload(_payload(selected, validation, performance, dominance, figure_dir))
