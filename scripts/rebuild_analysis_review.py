from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vix_regime_allocation.allocation import build_state_allocation
from vix_regime_allocation.backtest_plot import plot_four_portfolio_cumulative_performance
from vix_regime_allocation.backtest_summary import build_four_portfolio_performance_summary
from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.hmm_probability_plot import plot_hmm_smoothed_probabilities
from vix_regime_allocation.hmm_state_plot import plot_hmm_vix_states
from vix_regime_allocation.model_selection import build_hmm_model_comparison, select_preferred_hmm
from vix_regime_allocation.sensitivity import build_hmm_state_count_sensitivity
from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics
from vix_regime_allocation.strategy_comparison import build_dual_method_comparison

ROOT = Path(__file__).resolve().parents[1]
STEP1 = ROOT / "data/processed/step1_data.csv"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
GENERATED = ROOT / "reports/generated"


def _load_step1() -> pd.DataFrame:
    frame = pd.read_csv(STEP1, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def _parameter_table(fit: HMMFitResult) -> pd.DataFrame:
    states = fit.states.to_numpy(dtype=int)
    counts = np.bincount(states, minlength=fit.n_states)
    posterior_mean = fit.probabilities.mean(axis=0).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "state": np.arange(fit.n_states, dtype=int),
            "mean_vix_change": fit.means.to_numpy(dtype=float),
            "variance_vix_change": fit.variances.to_numpy(dtype=float),
            "start_probability": np.asarray(fit.start_probabilities, dtype=float),
            "viterbi_observations": counts,
            "viterbi_occupancy": counts.astype(float) / len(states),
            "posterior_mean_probability": posterior_mean,
        }
    )


def _write_csv(frame: pd.DataFrame, relative: str, *, index: bool = False) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def _write_states(states: pd.Series, relative: str) -> None:
    frame = states.astype("int64").rename("state").rename_axis("Date").reset_index()
    _write_csv(frame, relative)


def _write_json(payload: dict[str, object], relative: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hmm_candidates(data: pd.DataFrame) -> list[dict[str, object]]:
    vix_change = data["VIX_change"].rename("VIX_change")
    candidates = [evaluate_hmm_candidate(vix_change, n_states) for n_states in (2, 3)]
    for candidate in candidates:
        fit = candidate.get("fit")
        n_states = candidate.get("n_states")
        if not isinstance(fit, HMMFitResult) or not isinstance(n_states, int):
            raise RuntimeError("HMM evaluation returned an invalid candidate payload.")
        _write_csv(_parameter_table(fit), f"reports/tables/step2_hmm_{n_states}_parameters.csv")
        transition = fit.transition_matrix.rename_axis("from_state").reset_index()
        _write_csv(transition, f"reports/tables/step2_hmm_{n_states}_transition.csv")
        _write_states(fit.states, f"reports/tables/step2_hmm_{n_states}_states.csv")
    return candidates


def _steps_2_to_4(
    data: pd.DataFrame, candidates: list[dict[str, object]]
) -> tuple[pd.Series, pd.DataFrame, dict[int, pd.Series]]:
    comparison = build_hmm_model_comparison(candidates)
    selection = select_preferred_hmm(comparison, candidates)
    _write_csv(comparison, "reports/tables/step3_model_comparison.csv")

    selected_states = selection["states"]
    if not isinstance(selected_states, pd.Series):
        raise RuntimeError("HMM selection did not return a state path.")
    selected_states = selected_states.astype("int64").rename("state")
    _write_states(selected_states, "reports/tables/step3_selected_states.csv")

    statistics = compute_state_asset_statistics(data, selected_states)
    _write_csv(statistics, "reports/tables/step3_state_asset_statistics.csv")

    keep = build_state_allocation(statistics, "100_keep")
    spread = build_state_allocation(statistics, "60_40_spread")
    _write_csv(keep, "reports/tables/step4_allocation_100_keep.csv")
    _write_csv(spread, "reports/tables/step4_allocation_60_40_spread.csv")

    input_sha = hashlib.sha256(STEP1.read_bytes()).hexdigest()
    selected_payload: dict[str, object] = {
        "family": "hmm",
        "n_states": int(selection["n_states"]),
        "state_source": str(selection["state_source"]),
        "selection_reason": str(selection["selection_reason"]),
        "input_data_sha256": input_sha,
        "selected_states_path": "reports/tables/step3_selected_states.csv",
    }
    _write_json(selected_payload, "reports/generated/step3_selected_model.json")

    states_by_k: dict[int, pd.Series] = {}
    fits: dict[int, HMMFitResult] = {}
    for candidate in candidates:
        n_states = int(candidate["n_states"])
        fit = candidate.get("fit")
        if not isinstance(fit, HMMFitResult):
            raise RuntimeError("HMM candidate is missing its fit result.")
        fits[n_states] = fit
        states_by_k[n_states] = fit.states.astype("int64").rename("state")

    plot_hmm_vix_states(
        data["VIX"],
        states_by_k[2],
        states_by_k[3],
        FIGURES / "step2_hmm_vix_states.png",
    )
    plot_hmm_smoothed_probabilities(
        fits[2].probabilities,
        fits[3].probabilities,
        FIGURES / "step2_hmm_smoothed_probabilities.png",
    )
    plot_state_asset_statistics(statistics, FIGURES / "step3_state_asset_statistics.png")

    tables = [
        f"reports/tables/step2_hmm_{k}_{suffix}.csv"
        for k in (2, 3)
        for suffix in ("parameters", "states", "transition")
    ] + [
        "reports/tables/step3_model_comparison.csv",
        "reports/tables/step3_selected_states.csv",
        "reports/tables/step3_state_asset_statistics.csv",
        "reports/tables/step4_allocation_100_keep.csv",
        "reports/tables/step4_allocation_60_40_spread.csv",
    ]
    _write_json(
        {
            "schema_version": 2,
            "input_data_path": "data/processed/step1_data.csv",
            "input_data_sha256": input_sha,
            "selected_model_path": "reports/generated/step3_selected_model.json",
            "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
            "tables": sorted(tables),
            "figures": sorted(
                [
                    "reports/figures/step2_hmm_vix_states.png",
                    "reports/figures/step2_hmm_smoothed_probabilities.png",
                    "reports/figures/step3_state_asset_statistics.png",
                ]
            ),
        },
        "reports/generated/steps_2_4_manifest.json",
    )
    return selected_states, statistics, states_by_k


def _step_5(
    data: pd.DataFrame,
    selected_states: pd.Series,
    statistics: pd.DataFrame,
    states_by_k: dict[int, pd.Series],
) -> None:
    comparison, _ = build_dual_method_comparison(data, selected_states, statistics)
    _write_csv(comparison.rename_axis("Date").reset_index(), "reports/tables/step5_daily_returns.csv")

    summary = build_four_portfolio_performance_summary(comparison)
    _write_csv(summary, "reports/tables/step5_performance_summary.csv")

    sensitivity = build_hmm_state_count_sensitivity(data, states_by_k)
    _write_csv(sensitivity, "reports/tables/step5_state_count_sensitivity.csv")
    plot_four_portfolio_cumulative_performance(
        comparison, FIGURES / "step5_cumulative_performance.png"
    )

    input_sha = hashlib.sha256(STEP1.read_bytes()).hexdigest()
    tables = [
        "reports/tables/step5_daily_returns.csv",
        "reports/tables/step5_performance_summary.csv",
        "reports/tables/step5_state_count_sensitivity.csv",
    ]
    _write_json(
        {
            "schema_version": 2,
            "input_data_path": "data/processed/step1_data.csv",
            "input_data_sha256": input_sha,
            "selected_model_path": "reports/generated/step3_selected_model.json",
            "selected_states_path": "reports/tables/step3_selected_states.csv",
            "allocation_paths": [
                "reports/tables/step4_allocation_100_keep.csv",
                "reports/tables/step4_allocation_60_40_spread.csv",
            ],
            "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
            "tables": tables,
            "figures": ["reports/figures/step5_cumulative_performance.png"],
        },
        "reports/generated/step5_manifest.json",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    data = _load_step1()
    candidates = _hmm_candidates(data)
    selected_states, statistics, states_by_k = _steps_2_to_4(data, candidates)
    _step_5(data, selected_states, statistics, states_by_k)
    print("HMM-only dual-allocation analysis rebuild completed.")


if __name__ == "__main__":
    main()
