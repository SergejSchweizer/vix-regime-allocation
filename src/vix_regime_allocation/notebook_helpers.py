"""Notebook-facing orchestration helpers generated from the GWP2 notebook.

Implementation lives here so the notebook itself remains presentation-only.
"""

# ruff: noqa
# mypy: ignore-errors

from __future__ import annotations
from typing import Any


def data_source_and_financial_interpretation_002() -> Any:  # pragma: no cover
    """Notebook section: Data source and financial interpretation (cell 2)."""
    global \
        Image, \
        Markdown, \
        OUTPUT_COLUMNS, \
        Path, \
        TemporaryDirectory, \
        data, \
        data_path, \
        display, \
        np, \
        pd, \
        plot_etf_log_returns, \
        plot_vix_change, \
        re, \
        references_path, \
        repo_root
    import re
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import numpy as np
    import pandas as pd
    from IPython.display import Image, Markdown, display
    from vix_regime_allocation.plots import plot_etf_log_returns, plot_vix_change
    from vix_regime_allocation.transform import OUTPUT_COLUMNS

    repo_root = Path.cwd()
    if not (repo_root / "data/processed/step1_data.csv").is_file():
        repo_root = repo_root.parent
    data_path = repo_root / "data/processed/step1_data.csv"
    references_path = repo_root / "reports/references.bib"
    data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    data.index.name = "Date"
    assert tuple(data.columns) == OUTPUT_COLUMNS
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_monotonic_increasing
    assert not data.index.has_duplicates
    assert not data.isna().any().any()
    assert np.isfinite(data.to_numpy(dtype=float)).all()
    assert (data[["TLT", "GLD", "SPY", "VIX"]] > 0.0).all().all()
    return data.head()


def data_source_and_financial_interpretation_003() -> Any:  # pragma: no cover
    """Notebook section: Data source and financial interpretation (cell 3)."""
    global sample_summary
    sample_summary = pd.DataFrame(
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
    return sample_summary


def daily_change_in_vix_006() -> Any:  # pragma: no cover
    """Notebook section: Daily change in VIX (cell 6)."""
    global analysis_columns, descriptive_statistics
    analysis_columns = ["TLT_log_return", "GLD_log_return", "SPY_log_return", "VIX_change"]
    descriptive_statistics = data[analysis_columns].describe().T
    return descriptive_statistics


def daily_change_in_vix_007() -> Any:  # pragma: no cover
    """Notebook section: Daily change in VIX (cell 7)."""
    global etf_std, highest_std_series, highest_std_value, vix_extreme_change, vix_extreme_date
    vix_extreme_date = data["VIX_change"].abs().idxmax()
    vix_extreme_change = float(data.loc[vix_extreme_date, "VIX_change"])
    etf_std = data[["TLT_log_return", "GLD_log_return", "SPY_log_return"]].std(ddof=1)
    highest_std_series = str(etf_std.idxmax())
    highest_std_value = float(etf_std.loc[highest_std_series])
    return display(
        Markdown(
            f"**Computed Step 1 observations.** The common sample contains **{len(data)}** rows from **{data.index.min().date().isoformat()}** through **{data.index.max().date().isoformat()}**. The largest absolute daily VIX change occurs on **{vix_extreme_date.date().isoformat()}** and equals **{repr(vix_extreme_change)}** VIX points. Among the three ETF log-return series, **{highest_std_series}** has the largest sample daily standard deviation, **{repr(highest_std_value)}**, using `ddof=1`."
        )
    )


def required_etf_return_figure_009() -> Any:  # pragma: no cover
    """Notebook section: Required ETF-return figure (cell 9)."""
    global etf_plot_path, temporary_directory
    with TemporaryDirectory() as temporary_directory:
        etf_plot_path = Path(temporary_directory) / "step1_etf_log_returns.png"
        plot_etf_log_returns(data, etf_plot_path)
        display(Image(filename=str(etf_plot_path)))
    return None


def required_vix_change_figure_011() -> Any:  # pragma: no cover
    """Notebook section: Required VIX-change figure (cell 11)."""
    global temporary_directory, vix_plot_path
    with TemporaryDirectory() as temporary_directory:
        vix_plot_path = Path(temporary_directory) / "step1_vix_change.png"
        plot_vix_change(data, vix_plot_path)
        display(Image(filename=str(vix_plot_path)))
    return None


def step_2_modeling_vix_regimes_discrete_markov_chai_014() -> Any:  # pragma: no cover
    """Notebook section: Step 2 - Modeling VIX Regimes: Discrete Markov Chains (cell 14)."""
    global \
        candidate, \
        diagnostics, \
        evaluate_markov_candidate, \
        k, \
        markov_2, \
        markov_3, \
        markov_candidates, \
        markov_figure_dir, \
        markov_figure_path, \
        markov_output_dir, \
        plot_markov_vix_states
    from vix_regime_allocation.markov_evaluation import evaluate_markov_candidate
    from vix_regime_allocation.markov_plots import plot_markov_vix_states

    markov_output_dir = repo_root / "reports/tables"
    markov_figure_dir = repo_root / "reports/figures"
    markov_output_dir.mkdir(parents=True, exist_ok=True)
    markov_figure_dir.mkdir(parents=True, exist_ok=True)
    markov_2 = evaluate_markov_candidate(data["VIX_change"], 2)
    markov_3 = evaluate_markov_candidate(data["VIX_change"], 3)
    markov_candidates = {2: markov_2, 3: markov_3}
    for k, candidate in markov_candidates.items():
        candidate["thresholds"].to_csv(
            markov_output_dir / f"step2_markov_{k}_thresholds.csv", index=False
        )
        candidate["transition"].reset_index().to_csv(
            markov_output_dir / f"step2_markov_{k}_transition.csv", index=False
        )
        candidate["stationary"].reset_index().to_csv(
            markov_output_dir / f"step2_markov_{k}_stationary.csv", index=False
        )
        candidate["states"].rename_axis("Date").reset_index().to_csv(
            markov_output_dir / f"step2_markov_{k}_states.csv", index=False
        )
    markov_figure_path = markov_figure_dir / "step2_markov_vix_states.png"
    plot_markov_vix_states(data["VIX"], markov_2["states"], markov_3["states"], markov_figure_path)
    for k, candidate in markov_candidates.items():
        display(Markdown(f"### Markov candidate: K={k}"))
        display(Markdown("**Quantile intervals**"))
        display(candidate["thresholds"])
        display(Markdown("**Transition matrix**"))
        display(candidate["transition"])
        display(Markdown("**Stationary distribution**"))
        display(candidate["stationary"].to_frame())
        diagnostics = pd.Series(
            {
                "conditional_log_likelihood": candidate["log_likelihood"],
                "transition_observations": candidate["n_observations"],
                "free_transition_parameters": candidate["n_parameters"],
            },
            name="value",
        )
        display(diagnostics.to_frame())
    assert markov_2["states"].index.equals(data.index)
    assert markov_3["states"].index.equals(data.index)
    return Image(filename=str(markov_figure_path))


def step_2_modeling_vix_regimes_gaussian_hidden_mark_017() -> Any:  # pragma: no cover
    """Notebook section: Step 2 - Modeling VIX Regimes: Gaussian Hidden Markov Models (cell 17)."""
    global \
        _hmm_parameter_table, \
        candidate, \
        diagnostics, \
        evaluate_hmm_candidate, \
        fit, \
        hmm_2, \
        hmm_3, \
        hmm_candidates, \
        hmm_figure_dir, \
        hmm_output_dir, \
        hmm_probability_figure_path, \
        hmm_state_figure_path, \
        k, \
        parameters, \
        plot_hmm_smoothed_probabilities, \
        plot_hmm_vix_states
    from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
    from vix_regime_allocation.hmm_probability_plot import plot_hmm_smoothed_probabilities
    from vix_regime_allocation.hmm_state_plot import plot_hmm_vix_states

    hmm_output_dir = repo_root / "reports/tables"
    hmm_figure_dir = repo_root / "reports/figures"
    hmm_output_dir.mkdir(parents=True, exist_ok=True)
    hmm_figure_dir.mkdir(parents=True, exist_ok=True)
    hmm_2 = evaluate_hmm_candidate(data["VIX_change"], 2)
    hmm_3 = evaluate_hmm_candidate(data["VIX_change"], 3)
    hmm_candidates = {2: hmm_2, 3: hmm_3}

    def _hmm_parameter_table(fit):
        counts = fit.states.value_counts().reindex(range(fit.n_states), fill_value=0).astype(int)
        posterior_means = fit.probabilities.mean(axis=0)
        return pd.DataFrame(
            {
                "state": range(fit.n_states),
                "mean_vix_change": fit.means.to_numpy(dtype=float),
                "variance_vix_change": fit.variances.to_numpy(dtype=float),
                "start_probability": list(fit.start_probabilities),
                "viterbi_observations": counts.to_numpy(dtype=int),
                "viterbi_occupancy": counts.to_numpy(dtype=float) / len(fit.states),
                "posterior_mean_probability": [
                    float(posterior_means[f"state_{state}"]) for state in range(fit.n_states)
                ],
            }
        )

    for k, candidate in hmm_candidates.items():
        fit = candidate["fit"]
        parameters = _hmm_parameter_table(fit)
        parameters.to_csv(hmm_output_dir / f"step2_hmm_{k}_parameters.csv", index=False)
        fit.transition_matrix.reset_index().to_csv(
            hmm_output_dir / f"step2_hmm_{k}_transition.csv", index=False
        )
        fit.states.rename_axis("Date").reset_index().to_csv(
            hmm_output_dir / f"step2_hmm_{k}_states.csv", index=False
        )
        display(Markdown(f"### Gaussian HMM candidate: K={k}"))
        display(Markdown("**Fitted and decoded state parameters**"))
        display(parameters)
        display(Markdown("**Transition matrix**"))
        display(fit.transition_matrix)
        diagnostics = pd.Series(
            {
                "selected_restart_seed": fit.seed,
                "converged": fit.converged,
                "log_likelihood": candidate["log_likelihood"],
                "observations": candidate["n_observations"],
                "free_parameters": candidate["n_parameters"],
            },
            name="value",
        )
        display(diagnostics.to_frame())
    assert hmm_2["fit"].states.index.equals(data.index)
    assert hmm_3["fit"].states.index.equals(data.index)
    assert hmm_2["fit"].probabilities.index.equals(data.index)
    assert hmm_3["fit"].probabilities.index.equals(data.index)
    hmm_state_figure_path = hmm_figure_dir / "step2_hmm_vix_states.png"
    hmm_probability_figure_path = hmm_figure_dir / "step2_hmm_smoothed_probabilities.png"
    plot_hmm_vix_states(
        data["VIX"], hmm_2["fit"].states, hmm_3["fit"].states, hmm_state_figure_path
    )
    plot_hmm_smoothed_probabilities(
        hmm_2["fit"].probabilities, hmm_3["fit"].probabilities, hmm_probability_figure_path
    )
    display(Image(filename=str(hmm_state_figure_path)))
    return display(Image(filename=str(hmm_probability_figure_path)))


def step_3_model_selection_and_selected_state_proven_020() -> Any:  # pragma: no cover
    """Notebook section: Step 3 — Model selection and selected-state provenance (cell 20)."""
    global \
        COMPARISON_COLUMNS, \
        build_model_comparison, \
        comparison_path, \
        hmm_candidate_list, \
        input_sha256, \
        json, \
        k, \
        markov_candidate_list, \
        select_preferred_model, \
        selected_model, \
        selected_model_path, \
        selected_states, \
        selected_states_path, \
        sha256, \
        source_states, \
        state_source_path, \
        step3_comparison, \
        step3_generated_dir, \
        step3_selection, \
        step3_table_dir
    import json
    from hashlib import sha256
    from vix_regime_allocation.model_selection import (
        COMPARISON_COLUMNS,
        build_model_comparison,
        select_preferred_model,
    )

    step3_table_dir = repo_root / "reports/tables"
    step3_generated_dir = repo_root / "reports/generated"
    step3_table_dir.mkdir(parents=True, exist_ok=True)
    step3_generated_dir.mkdir(parents=True, exist_ok=True)
    markov_candidate_list = [markov_candidates[k] for k in (2, 3)]
    hmm_candidate_list = [hmm_candidates[k] for k in (2, 3)]
    step3_comparison = build_model_comparison(markov_candidate_list, hmm_candidate_list)
    step3_selection = select_preferred_model(
        step3_comparison, markov_candidate_list, hmm_candidate_list
    )
    comparison_path = step3_table_dir / "step3_model_comparison.csv"
    selected_states_path = step3_table_dir / "step3_selected_states.csv"
    selected_model_path = step3_generated_dir / "step3_selected_model.json"
    assert tuple(step3_comparison.columns) == COMPARISON_COLUMNS
    assert len(step3_comparison) == 4
    step3_comparison.to_csv(comparison_path, index=False)
    selected_states = step3_selection["states"].copy()
    selected_states.name = "state"
    assert selected_states.index.equals(data.index)
    assert selected_states.index.name == "Date"
    assert selected_states.notna().all()
    state_source_path = repo_root / str(step3_selection["state_source"])
    source_states = pd.read_csv(state_source_path, parse_dates=["Date"], index_col="Date")["state"]
    source_states.index.name = "Date"
    source_states.name = "state"
    source_states = source_states.astype("int64")
    selected_states = selected_states.astype("int64")
    assert source_states.equals(selected_states)
    selected_states.rename_axis("Date").reset_index().to_csv(selected_states_path, index=False)
    input_sha256 = sha256(data_path.read_bytes()).hexdigest()
    selected_model = {
        "family": step3_selection["family"],
        "n_states": int(step3_selection["n_states"]),
        "state_source": step3_selection["state_source"],
        "selection_reason": step3_selection["selection_reason"],
        "markov_best_n_states": int(step3_selection["markov_best_n_states"]),
        "hmm_best_n_states": int(step3_selection["hmm_best_n_states"]),
        "input_data_sha256": input_sha256,
        "selected_states_path": "reports/tables/step3_selected_states.csv",
    }
    assert tuple(selected_model) == (
        "family",
        "n_states",
        "state_source",
        "selection_reason",
        "markov_best_n_states",
        "hmm_best_n_states",
        "input_data_sha256",
        "selected_states_path",
    )
    selected_model_path.write_text(json.dumps(selected_model, indent=2) + chr(10), encoding="utf-8")
    display(Markdown("### Four-candidate information-criterion table"))
    display(step3_comparison)
    display(Markdown("### Deterministic preferred-model decision"))
    display(pd.Series(selected_model, name="value").to_frame())
    return display(
        Markdown(
            f"**Selected model:** {selected_model['family'].upper()} with $K={selected_model['n_states']}$. {selected_model['selection_reason']}"
        )
    )


def step_3_state_conditional_etf_analysis_023() -> Any:  # pragma: no cover
    """Notebook section: Step 3 — State-conditional ETF analysis (cell 23)."""
    global \
        compute_state_asset_statistics, \
        expected_rows, \
        plot_state_asset_statistics, \
        selected_state_frame, \
        selected_states_for_statistics, \
        selected_states_path, \
        state_asset_statistics, \
        state_leaders, \
        statistics_figure_path, \
        statistics_path
    from vix_regime_allocation.state_statistics import compute_state_asset_statistics
    from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics

    selected_states_path = repo_root / "reports/tables/step3_selected_states.csv"
    statistics_path = repo_root / "reports/tables/step3_state_asset_statistics.csv"
    statistics_figure_path = repo_root / "reports/figures/step3_state_asset_statistics.png"
    selected_state_frame = pd.read_csv(selected_states_path, parse_dates=["Date"], index_col="Date")
    selected_state_frame.index.name = "Date"
    selected_states_for_statistics = selected_state_frame["state"].astype("int64")
    selected_states_for_statistics.name = "state"
    assert selected_states_for_statistics.index.equals(data.index)
    assert selected_states_for_statistics.notna().all()
    state_asset_statistics = compute_state_asset_statistics(data, selected_states_for_statistics)
    state_asset_statistics.to_csv(statistics_path, index=False)
    plot_state_asset_statistics(state_asset_statistics, statistics_figure_path)
    expected_rows = 3 * int(selected_states_for_statistics.nunique())
    assert len(state_asset_statistics) == expected_rows
    assert state_asset_statistics["observations"].gt(0).all()
    assert statistics_figure_path.is_file() and statistics_figure_path.stat().st_size > 0
    display(Markdown("### State-conditional ETF return statistics"))
    display(state_asset_statistics)
    display(Image(filename=str(statistics_figure_path)))
    state_leaders = (
        state_asset_statistics.sort_values(
            ["state", "mean_log_return", "asset"], ascending=[True, False, True], kind="stable"
        )
        .groupby("state", sort=True, as_index=False)
        .first()[["state", "asset", "mean_log_return", "std_log_return", "observations"]]
    )
    display(Markdown("### Highest conditional mean daily log return in each state"))
    return display(state_leaders)


def step_4_state_based_rotation_rule_026() -> Any:  # pragma: no cover
    """Notebook section: Step 4 — State-based rotation rule (cell 26)."""
    global \
        ALLOCATION_COLUMNS, \
        build_state_allocation, \
        row, \
        step3_statistics, \
        step3_statistics_path, \
        step4_allocation, \
        step4_allocation_path
    from vix_regime_allocation.allocation import ALLOCATION_COLUMNS, build_state_allocation

    step3_statistics_path = repo_root / "reports/tables/step3_state_asset_statistics.csv"
    step4_allocation_path = repo_root / "reports/tables/step4_allocation_mapping.csv"
    step3_statistics = pd.read_csv(step3_statistics_path)
    step4_allocation = build_state_allocation(step3_statistics)
    assert tuple(step4_allocation.columns) == ALLOCATION_COLUMNS
    assert np.allclose(
        step4_allocation[["TLT_weight", "GLD_weight", "SPY_weight"]].sum(axis=1), 1.0
    )
    for row in step4_allocation.itertuples(index=False):
        assert getattr(row, f"{row.selected_asset}_weight") == 1.0
    step4_allocation.to_csv(step4_allocation_path, index=False)
    display(Markdown("### Canonical state-to-allocation mapping"))
    return display(step4_allocation)


def deterministic_steps_2_4_artifact_manifest_029() -> Any:  # pragma: no cover
    """Notebook section: Deterministic Steps 2–4 artifact manifest (cell 29)."""
    global \
        figures, \
        hashlib, \
        input_sha256, \
        json, \
        manifest, \
        manifest_path, \
        relative_path, \
        selected_model, \
        selected_model_path, \
        step1_path, \
        tables
    import hashlib
    import json

    step1_path = repo_root / "data/processed/step1_data.csv"
    selected_model_path = repo_root / "reports/generated/step3_selected_model.json"
    manifest_path = repo_root / "reports/generated/steps_2_4_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    input_sha256 = hashlib.sha256(step1_path.read_bytes()).hexdigest()
    selected_model = json.loads(selected_model_path.read_text(encoding="utf-8"))
    assert selected_model["input_data_sha256"] == input_sha256
    assert selected_model["selected_states_path"] == "reports/tables/step3_selected_states.csv"
    tables = sorted(
        [
            "reports/tables/step2_hmm_2_parameters.csv",
            "reports/tables/step2_hmm_2_states.csv",
            "reports/tables/step2_hmm_2_transition.csv",
            "reports/tables/step2_hmm_3_parameters.csv",
            "reports/tables/step2_hmm_3_states.csv",
            "reports/tables/step2_hmm_3_transition.csv",
            "reports/tables/step2_markov_2_states.csv",
            "reports/tables/step2_markov_2_stationary.csv",
            "reports/tables/step2_markov_2_thresholds.csv",
            "reports/tables/step2_markov_2_transition.csv",
            "reports/tables/step2_markov_3_states.csv",
            "reports/tables/step2_markov_3_stationary.csv",
            "reports/tables/step2_markov_3_thresholds.csv",
            "reports/tables/step2_markov_3_transition.csv",
            "reports/tables/step3_model_comparison.csv",
            "reports/tables/step3_selected_states.csv",
            "reports/tables/step3_state_asset_statistics.csv",
            "reports/tables/step4_allocation_mapping.csv",
        ]
    )
    figures = sorted(
        [
            "reports/figures/step2_hmm_smoothed_probabilities.png",
            "reports/figures/step2_hmm_vix_states.png",
            "reports/figures/step2_markov_vix_states.png",
            "reports/figures/step3_state_asset_statistics.png",
        ]
    )
    for relative_path in tables + figures:
        assert (repo_root / relative_path).is_file(), relative_path
    manifest = {
        "schema_version": 1,
        "input_data_path": "data/processed/step1_data.csv",
        "input_data_sha256": input_sha256,
        "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
        "selected_model_path": "reports/generated/step3_selected_model.json",
        "tables": tables,
        "figures": figures,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return display(Markdown("```json\n" + json.dumps(manifest, indent=2, sort_keys=True) + "\n```"))


def step_5_backtest_construction_and_required_benchm_031() -> Any:  # pragma: no cover
    """Notebook section: Step 5 — Backtest construction and required benchmarks (cell 31)."""
    global \
        COMPARISON_COLUMNS, \
        Path, \
        ROTATION_DETAIL_COLUMNS, \
        allocation, \
        allocation_path, \
        build_comparison, \
        build_equal_weight_monthly_returns, \
        build_rotation_returns, \
        build_spy_buy_hold_returns, \
        comparison_index, \
        daily_returns_path, \
        equal_weight, \
        hashlib, \
        input_sha256, \
        json, \
        repo_root, \
        rotation_detail, \
        selected_model, \
        selected_model_path, \
        selected_states, \
        selected_states_frame, \
        selected_states_path, \
        spy_buy_hold, \
        step1, \
        step1_path, \
        step5_daily_returns
    import hashlib
    import json
    from pathlib import Path
    from vix_regime_allocation.backtest import ROTATION_DETAIL_COLUMNS, build_rotation_returns
    from vix_regime_allocation.backtest_summary import COMPARISON_COLUMNS, build_comparison
    from vix_regime_allocation.benchmarks import (
        build_equal_weight_monthly_returns,
        build_spy_buy_hold_returns,
    )

    repo_root = (
        Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
    )
    step1_path = repo_root / "data/processed/step1_data.csv"
    selected_model_path = repo_root / "reports/generated/step3_selected_model.json"
    allocation_path = repo_root / "reports/tables/step4_allocation_mapping.csv"
    daily_returns_path = repo_root / "reports/tables/step5_daily_returns.csv"
    step1 = pd.read_csv(step1_path, parse_dates=["Date"]).set_index("Date")
    step1.index = pd.DatetimeIndex(step1.index, name="Date")
    selected_model = json.loads(selected_model_path.read_text(encoding="utf-8"))
    input_sha256 = hashlib.sha256(step1_path.read_bytes()).hexdigest()
    assert selected_model["input_data_sha256"] == input_sha256
    selected_states_path = repo_root / selected_model["selected_states_path"]
    selected_states_frame = pd.read_csv(selected_states_path, parse_dates=["Date"])
    assert selected_states_frame["Date"].tolist() == step1.index.tolist()
    selected_states = pd.Series(
        selected_states_frame["state"].to_numpy(dtype=int),
        index=step1.index,
        name="state",
        dtype=int,
    )
    allocation = pd.read_csv(allocation_path)
    rotation_detail = build_rotation_returns(step1, selected_states, allocation)
    assert tuple(rotation_detail.columns) == ROTATION_DETAIL_COLUMNS
    comparison_index = pd.DatetimeIndex(rotation_detail.index, name="Date")
    equal_weight = build_equal_weight_monthly_returns(step1, comparison_index)
    spy_buy_hold = build_spy_buy_hold_returns(step1, comparison_index)
    step5_daily_returns = build_comparison(rotation_detail, equal_weight, spy_buy_hold)
    assert tuple(step5_daily_returns.columns) == COMPARISON_COLUMNS
    assert step5_daily_returns.index.equals(comparison_index)
    assert len(step5_daily_returns) == len(step1) - 1
    daily_returns_path.parent.mkdir(parents=True, exist_ok=True)
    step5_daily_returns.to_csv(daily_returns_path, index_label="Date")
    display(Markdown("### Lagged rotation decision examples"))
    display(rotation_detail.head(8))
    display(Markdown("### Canonical Step 5 daily comparison returns"))
    display(step5_daily_returns.head(8))
    return display(
        Markdown(
            f"Comparison period: **{comparison_index[0].date()}** to **{comparison_index[-1].date()}**, observations: **{len(comparison_index)}**."
        )
    )


def step_5_performance_metrics_and_cumulative_compar_034() -> Any:  # pragma: no cover
    """Notebook section: Step 5 — Performance metrics and cumulative comparison (cell 34)."""
    global \
        build_performance_summary, \
        daily_path, \
        figure_path, \
        label, \
        lines, \
        performance_summary, \
        plot_cumulative_performance, \
        portfolio, \
        repo_root_step5, \
        row, \
        rows, \
        step5_daily, \
        summary_path
    from vix_regime_allocation.backtest_plot import plot_cumulative_performance
    from vix_regime_allocation.backtest_summary import build_performance_summary

    if Path.cwd().name == "notebooks":
        repo_root_step5 = Path.cwd().resolve().parent
    else:
        repo_root_step5 = Path.cwd().resolve()
    daily_path = repo_root_step5 / "reports/tables/step5_daily_returns.csv"
    summary_path = repo_root_step5 / "reports/tables/step5_performance_summary.csv"
    figure_path = repo_root_step5 / "reports/figures/step5_cumulative_performance.png"
    step5_daily = pd.read_csv(daily_path, parse_dates=["Date"]).set_index("Date")
    step5_daily.index = pd.DatetimeIndex(step5_daily.index, name="Date")
    performance_summary = build_performance_summary(step5_daily)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    performance_summary.to_csv(summary_path, index=False)
    plot_cumulative_performance(step5_daily, figure_path)
    display(Markdown("### Required performance summary"))
    display(performance_summary)
    display(Image(filename=str(figure_path)))
    rows = performance_summary.set_index("portfolio")
    lines = []
    for portfolio, label in [
        ("regime_rotation", "Regime rotation"),
        ("equal_weight_monthly", "Equal-weight monthly reset"),
        ("spy_buy_hold", "SPY buy and hold"),
    ]:
        row = rows.loc[portfolio]
        lines.append(
            f"- **{label}:** cumulative return {row['cumulative_return']:.6f}; annualized return {row['annualized_return']:.6f}; annualized volatility {row['annualized_volatility']:.6f}; Sharpe {row['sharpe_ratio']:.6f}; maximum drawdown {row['max_drawdown']:.6f}; observations {int(row['observations'])}."
        )
    return display(Markdown("### Numerical comparison\n" + "\n".join(lines)))


def works_cited_037() -> Any:  # pragma: no cover
    """Notebook section: Works Cited (cell 37)."""
    global \
        _mla_author, \
        _mla_entry, \
        _parse_bibtex_registry, \
        cited_keys, \
        key, \
        missing_keys, \
        references, \
        references_path
    references_path = repo_root / "reports/references.bib"

    def _parse_bibtex_registry(text: str) -> dict[str, dict[str, str]]:
        entries: dict[str, dict[str, str]] = {}
        for match in re.finditer("@(\\w+)\\{([^,]+),\\s*(.*?)\\n\\}", text, flags=re.DOTALL):
            entry_type, key, body = match.groups()
            fields = {field: value for field, value in re.findall('(\\w+)\\s*=\\s*"([^"]*)"', body)}
            fields["entry_type"] = entry_type.lower()
            entries[key] = fields
        return entries

    def _mla_author(author: str) -> str:
        authors = author.split(" and ")
        return authors[0] + (", et al." if len(authors) > 2 else "")

    def _mla_entry(entry: dict[str, str]) -> str:
        author = _mla_author(entry["author"])
        if entry["entry_type"] == "article":
            pages = entry["pages"].replace("--", "-")
            return f'''{author}. "{entry["title"]}." *{entry["journal"]}*, vol. {entry["volume"]}, no. {entry["number"]}, {entry["year"]}, pp. {pages}. doi:{entry["doi"]}.'''
        return f'''{author}. "{entry["title"]}." *{entry["publisher"]}*, {entry["year"]}, {entry["url"]}. Accessed 19 Aug. 2026.'''

    references = _parse_bibtex_registry(references_path.read_text(encoding="utf-8"))
    cited_keys = [
        "whaley2009vix",
        "cboe2019vixfaq",
        "baum1970maximization",
        "rabiner1989tutorial",
        "viterbi1967decoding",
        "akaike1974identification",
        "schwarz1978dimension",
        "markowitz1952portfolio",
        "white2000datasnooping",
        "bailey2014deflatedsharpe",
    ]
    missing_keys = [key for key in cited_keys if key not in references]
    assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
    return display(
        Markdown(chr(10).join((f"- {_mla_entry(references[key])}" for key in cited_keys)))
    )
