from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd

from vix_regime_allocation.backtest_plot import plot_cumulative_performance
from vix_regime_allocation.hmm_probability_plot import plot_hmm_smoothed_probabilities
from vix_regime_allocation.hmm_state_plot import plot_hmm_vix_states
from vix_regime_allocation.markov_plots import plot_markov_vix_states
from vix_regime_allocation.plots import plot_etf_log_returns, plot_vix_change
from vix_regime_allocation.sensitivity import build_state_count_sensitivity
from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
STEP1 = ROOT / "data/processed/step1_data.csv"
SELECTED = ROOT / "reports/generated/step3_selected_model.json"
STATE_STATS = ROOT / "reports/tables/step3_state_asset_statistics.csv"
DAILY = ROOT / "reports/tables/step5_daily_returns.csv"
OUT_SENSITIVITY = ROOT / "reports/tables/step5_state_count_sensitivity.csv"
OUT_MANIFEST = ROOT / "reports/generated/step5_manifest.json"


def _load_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one notebook occurrence, found {count}: {old[:70]!r}")
    return text.replace(old, new, 1)


def _update_notebook_markdown() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    greek_old = (
        "**Greek letters used in the emission equation:** μ — *mu*, pronounced “mew”; "
        "σ — *sigma*, pronounced “SIG-muh”."
    )
    greek_new = (
        "**Greek letters used in the emission equation:** Δ — *delta*, pronounced “DEL-tuh”; "
        "μ — *mu*, pronounced “mew”; σ — *sigma*, pronounced “SIG-muh”."
    )
    greek_hits = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown" and greek_old in cell.source:
            cell.source = cell.source.replace(greek_old, greek_new)
            greek_hits += 1
    if greek_hits != 1:
        raise RuntimeError(f"Expected one HMM Greek-letter declaration, found {greek_hits}.")

    old_plot_text = (
        "The reported quantities remain on a **daily log-return scale**; they are not "
        "annualized. The figure shows conditional mean daily log returns as bars and uses "
        "the corresponding **sample standard deviation** as the error-bar magnitude. Those "
        "error bars describe within-state return dispersion; they are not standard errors "
        "and are not confidence intervals."
    )
    new_plot_text = (
        "The reported quantities remain on a **daily log-return scale**; they are not "
        "annualized. For readability the figure converts both quantities to daily basis "
        "points. The upper panel shows conditional mean daily log returns; the lower panel "
        "shows the corresponding **sample standard deviation**. Keeping dispersion in a "
        "separate panel avoids the common visual mistake of drawing a sample standard "
        "deviation as if it were an error bar for uncertainty in the mean. The lower panel "
        "describes within-state return dispersion; it is not a standard error or confidence "
        "interval."
    )
    plot_hits = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown" and old_plot_text in cell.source:
            cell.source = cell.source.replace(old_plot_text, new_plot_text)
            plot_hits += 1
    if plot_hits != 1:
        raise RuntimeError(f"Expected one Step 3 plot explanation, found {plot_hits}.")

    selection_anchor = (
        "The comparison table intentionally reports all four candidates in one place for "
        "transparency, but `criterion_scope = within_family_only` prevents the table from "
        "implying that the smallest raw AIC or BIC across unlike likelihood constructions "
        "is the winner. The two state-count decisions are therefore made separately inside "
        "the Markov and HMM families before the deterministic method-validity rule is applied."
    )
    selection_addition = selection_anchor + (
        "\n\nThe HMM $K=3$ candidate is a near-boundary case under that project validity rule: "
        "its least-populated Viterbi state contains exactly 259 of 5,465 observations, "
        "or 4.739249771271729% of the sample. This is only 0.260750228728271 percentage "
        "points below the fixed 5% occupancy threshold. The fallback to Markov $K=2$ is "
        "therefore a deterministic project-governance choice for avoiding a very small "
        "decoded state, not a formal statistical proof that the HMM specification is invalid."
    )
    selection_hits = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown" and selection_anchor in cell.source:
            cell.source = cell.source.replace(selection_anchor, selection_addition)
            selection_hits += 1
    if selection_hits != 1:
        raise RuntimeError(f"Expected one model-selection explanation, found {selection_hits}.")

    heading = "### Step 5 — State-count sensitivity and predictive interpretation"
    works_index = next(
        (
            index
            for index, cell in enumerate(nb.cells)
            if cell.cell_type == "markdown" and "## Works Cited" in cell.source
        ),
        None,
    )
    if works_index is None:
        raise RuntimeError("Works Cited section not found.")
    old_index = next(
        (
            index
            for index, cell in enumerate(nb.cells)
            if cell.cell_type == "markdown" and heading in cell.source
        ),
        None,
    )
    if old_index is not None:
        nb.cells = nb.cells[:old_index] + nb.cells[works_index:]
        works_index = old_index

    sensitivity_md = nbformat.v4.new_markdown_cell(
        """### Step 5 — State-count sensitivity and predictive interpretation

The final sensitivity check compares exactly two and three states **within the already selected model family**. It reuses the canonical persisted state paths and delegates the state-conditional ETF statistics, deterministic allocation map, one-observation execution lag, and performance metrics to the shared project functions. No regime model is refitted and no state path is redecoded in this sensitivity step.

**Greek letters used in this section:** none.

A key interpretation distinction is now explicit. Step 3 estimates a **contemporaneous association**: it asks which ETF had the highest return on dates classified into a given daily-ΔVIX state. Step 5 is harder: the state observed at date $t-1$ chooses the asset whose return is realized at date $t$. The backtest therefore tests whether a same-day state/return relationship contains useful **next-observation predictive information**. A strong contemporaneous separation need not survive that one-row lag.

The two sensitivity rows use the common lagged return-date intersection, so cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and observation count are compared on identical dates."""
    )
    sensitivity_code = nbformat.v4.new_code_cell(
        r'''from vix_regime_allocation.sensitivity import build_state_count_sensitivity

preferred_family = selected_model["family"]
states_by_k = {}
for n_states in (2, 3):
    states_path = (
        repo_root / f"reports/tables/step2_{preferred_family}_{n_states}_states.csv"
    )
    state_frame = pd.read_csv(states_path, parse_dates=["Date"]).set_index("Date")
    state_frame.index = pd.DatetimeIndex(state_frame.index, name="Date")
    states_by_k[n_states] = state_frame["state"].astype(int).rename("state")

state_count_sensitivity = build_state_count_sensitivity(
    step1_data,
    preferred_family,
    states_by_k,
)
sensitivity_path = repo_root / "reports/tables/step5_state_count_sensitivity.csv"
state_count_sensitivity.to_csv(sensitivity_path, index=False)
display(Markdown("### K=2 versus K=3 sensitivity within the preferred family"))
display(state_count_sensitivity)'''
    )
    final_md = nbformat.v4.new_markdown_cell(
        """### Final Step 5 interpretation

The required one-observation lag prevents an impossible same-row trade, but it does **not** remove all look-ahead. Regime thresholds/model parameters, the selected state path, and the state-to-ETF mapping were estimated using the full historical sample. The reported performance is therefore descriptive and in-sample rather than a causal or out-of-sample estimate (White; Bailey and Lopez de Prado). <!-- citekey: white2000datasnooping --><!-- citekey: bailey2014deflatedsharpe -->

The primary two-state rotation is not economically competitive with either required benchmark on this sample. It finishes with lower cumulative and annualized return than both benchmarks, has a lower Sharpe ratio than both, and experiences substantially more drawdown and volatility than the monthly equal-weight benchmark. Its maximum drawdown is only modestly smaller in magnitude than SPY's while its return is much lower. This is consistent with the central predictive lesson above: the same-day ΔVIX state/ETF-return pattern used to construct the mapping does not translate into a sufficiently strong next-observation signal after the required lag.

The state-count table is a robustness description rather than a second round of model selection. A production-quality extension should use rolling or expanding estimation, one-sided state inference, allocation means estimated only from information available at each decision date, and explicit turnover and transaction costs. Those changes would directly test whether any predictive value remains when the principal full-sample look-ahead channels are removed.

**Source note.** Step 5 returns, metrics, benchmark values, and state-count sensitivity: project team calculations from the canonical artifacts. Backtest-overfitting and data-snooping cautions: White (2000) and Bailey and Lopez de Prado (2014)."""
    )
    nb.cells[works_index:works_index] = [sensitivity_md, sensitivity_code, final_md]
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)


def _write_sensitivity_and_manifest() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    family = selected["family"]
    if family not in {"markov", "hmm"}:
        raise RuntimeError(f"Unexpected preferred family: {family!r}")

    step1 = _load_indexed_csv(STEP1)
    states_by_k: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        path = ROOT / f"reports/tables/step2_{family}_{n_states}_states.csv"
        frame = _load_indexed_csv(path)
        states_by_k[n_states] = frame["state"].astype(int).rename("state")
    sensitivity = build_state_count_sensitivity(step1, family, states_by_k)
    OUT_SENSITIVITY.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)

    tables = [
        "reports/tables/step5_daily_returns.csv",
        "reports/tables/step5_performance_summary.csv",
        "reports/tables/step5_state_count_sensitivity.csv",
    ]
    figures = ["reports/figures/step5_cumulative_performance.png"]
    for relative in tables + figures:
        path = ROOT / relative
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing canonical Step 5 artifact: {relative}")
    manifest = {
        "schema_version": 1,
        "input_data_path": "data/processed/step1_data.csv",
        "input_data_sha256": hashlib.sha256(STEP1.read_bytes()).hexdigest(),
        "selected_model_path": "reports/generated/step3_selected_model.json",
        "selected_states_path": selected["selected_states_path"],
        "allocation_path": "reports/tables/step4_allocation_mapping.csv",
        "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
        "tables": sorted(tables),
        "figures": sorted(figures),
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _regenerate_figures() -> None:
    step1 = _load_indexed_csv(STEP1)
    plot_etf_log_returns(step1, ROOT / "reports/figures/step1_etf_log_returns.png")
    plot_vix_change(step1, ROOT / "reports/figures/step1_vix_change.png")

    vix = step1["VIX"].rename("VIX")
    markov_2 = _load_indexed_csv(ROOT / "reports/tables/step2_markov_2_states.csv")[
        "state"
    ].astype(int).rename("state")
    markov_3 = _load_indexed_csv(ROOT / "reports/tables/step2_markov_3_states.csv")[
        "state"
    ].astype(int).rename("state")
    plot_markov_vix_states(
        vix,
        markov_2,
        markov_3,
        ROOT / "reports/figures/step2_markov_vix_states.png",
    )

    hmm_2 = _load_indexed_csv(ROOT / "reports/tables/step2_hmm_2_states.csv")["state"].astype(
        int
    ).rename("state")
    hmm_3 = _load_indexed_csv(ROOT / "reports/tables/step2_hmm_3_states.csv")["state"].astype(
        int
    ).rename("state")
    plot_hmm_vix_states(
        vix,
        hmm_2,
        hmm_3,
        ROOT / "reports/figures/step2_hmm_vix_states.png",
    )

    probabilities_2 = _load_indexed_csv(
        ROOT / "reports/tables/step2_hmm_2_probabilities.csv"
    )
    probabilities_3 = _load_indexed_csv(
        ROOT / "reports/tables/step2_hmm_3_probabilities.csv"
    )
    plot_hmm_smoothed_probabilities(
        probabilities_2,
        probabilities_3,
        ROOT / "reports/figures/step2_hmm_smoothed_probabilities.png",
    )

    statistics = pd.read_csv(STATE_STATS)
    plot_state_asset_statistics(
        statistics,
        ROOT / "reports/figures/step3_state_asset_statistics.png",
    )
    daily = _load_indexed_csv(DAILY)
    plot_cumulative_performance(
        daily,
        ROOT / "reports/figures/step5_cumulative_performance.png",
    )


def main() -> None:
    _regenerate_figures()
    _write_sensitivity_and_manifest()
    _update_notebook_markdown()
    print("Analysis review rebuild completed.")


if __name__ == "__main__":
    main()
