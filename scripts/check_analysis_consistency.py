from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vix_regime_allocation.allocation import build_state_allocation
from vix_regime_allocation.backtest import build_rotation_returns
from vix_regime_allocation.backtest_summary import build_comparison, build_performance_summary
from vix_regime_allocation.benchmarks import (
    build_equal_weight_monthly_returns,
    build_spy_buy_hold_returns,
)
from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.markov_evaluation import evaluate_markov_candidate
from vix_regime_allocation.model_selection import build_model_comparison, select_preferred_model
from vix_regime_allocation.sensitivity import build_state_count_sensitivity
from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.transform import OUTPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-9
ATOL = 1e-11


def _indexed_csv(relative: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / relative, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def _states(relative: str) -> pd.Series:
    frame = _indexed_csv(relative)
    if list(frame.columns) != ["state"]:
        raise AssertionError(f"{relative}: expected only a state column")
    return frame["state"].astype("int64").rename("state")


def _assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise AssertionError(f"{label} mismatch: {exc}") from exc


def _assert_series(actual: pd.Series, expected: pd.Series, label: str) -> None:
    try:
        pd.testing.assert_series_equal(
            actual,
            expected,
            check_exact=False,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise AssertionError(f"{label} mismatch: {exc}") from exc


def _step1() -> pd.DataFrame:
    data = _indexed_csv("data/processed/step1_data.csv")
    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Step 1 schema is not canonical")
    if len(data) < 2 or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise AssertionError("Step 1 Date index is invalid")
    values = data.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise AssertionError("Step 1 contains non-finite values")
    if np.any(data[["TLT", "GLD", "SPY", "VIX"]].to_numpy(dtype=float) <= 0.0):
        raise AssertionError("Step 1 contains non-positive price/index levels")

    # The first stored return needs the one dropped pre-sample price and cannot be reconstructed
    # from the processed CSV alone. Every subsequent stored row can and must reconcile exactly.
    for asset in ("TLT", "GLD", "SPY"):
        expected = np.log(
            data[asset].iloc[1:].to_numpy(dtype=float)
            / data[asset].iloc[:-1].to_numpy(dtype=float)
        )
        actual = data[f"{asset}_log_return"].iloc[1:].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    expected_vix_change = (
        data["VIX"].iloc[1:].to_numpy(dtype=float)
        - data["VIX"].iloc[:-1].to_numpy(dtype=float)
    )
    np.testing.assert_allclose(
        data["VIX_change"].iloc[1:].to_numpy(dtype=float),
        expected_vix_change,
        rtol=RTOL,
        atol=ATOL,
    )
    return data


def _persisted_hmm_parameters(n_states: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"reports/tables/step2_hmm_{n_states}_parameters.csv")


def _computed_hmm_parameters(candidate: dict[str, object]) -> pd.DataFrame:
    fit = candidate["fit"]
    states = fit.states.to_numpy(dtype=int)  # type: ignore[union-attr]
    counts = np.bincount(states, minlength=fit.n_states)  # type: ignore[union-attr]
    occupancy = counts.astype(float) / len(states)
    posterior_mean = fit.probabilities.mean(axis=0).to_numpy(dtype=float)  # type: ignore[union-attr]
    return pd.DataFrame(
        {
            "state": np.arange(fit.n_states, dtype=int),  # type: ignore[union-attr]
            "mean_vix_change": fit.means.to_numpy(dtype=float),  # type: ignore[union-attr]
            "variance_vix_change": fit.variances.to_numpy(dtype=float),  # type: ignore[union-attr]
            "start_probability": np.asarray(fit.start_probabilities, dtype=float),  # type: ignore[union-attr]
            "viterbi_observations": counts,
            "viterbi_occupancy": occupancy,
            "posterior_mean_probability": posterior_mean,
        }
    )


def _verify_regime_models(data: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    vix_change = data["VIX_change"].rename("VIX_change")
    markov_candidates: list[dict[str, object]] = []
    hmm_candidates: list[dict[str, object]] = []

    for n_states in (2, 3):
        candidate = evaluate_markov_candidate(vix_change, n_states)
        markov_candidates.append(candidate)
        _assert_series(
            candidate["states"],  # type: ignore[arg-type]
            _states(f"reports/tables/step2_markov_{n_states}_states.csv"),
            f"Markov K={n_states} states",
        )

        thresholds = pd.read_csv(ROOT / f"reports/tables/step2_markov_{n_states}_thresholds.csv")
        _assert_frame(candidate["thresholds"], thresholds, f"Markov K={n_states} thresholds")  # type: ignore[arg-type]

        transition = pd.read_csv(
            ROOT / f"reports/tables/step2_markov_{n_states}_transition.csv"
        ).set_index("from_state")
        transition.index.name = "from_state"
        _assert_frame(candidate["transition"], transition, f"Markov K={n_states} transition")  # type: ignore[arg-type]

        stationary = pd.read_csv(
            ROOT / f"reports/tables/step2_markov_{n_states}_stationary.csv"
        ).set_index("state")["stationary_probability"]
        stationary.index.name = "state"
        stationary.name = "stationary_probability"
        _assert_series(candidate["stationary"], stationary, f"Markov K={n_states} stationary")  # type: ignore[arg-type]

    for n_states in (2, 3):
        candidate = evaluate_hmm_candidate(vix_change, n_states)
        hmm_candidates.append(candidate)
        fit = candidate["fit"]
        _assert_series(
            fit.states,  # type: ignore[union-attr]
            _states(f"reports/tables/step2_hmm_{n_states}_states.csv"),
            f"HMM K={n_states} Viterbi states",
        )
        transition = pd.read_csv(
            ROOT / f"reports/tables/step2_hmm_{n_states}_transition.csv"
        ).set_index("from_state")
        transition.index.name = "from_state"
        _assert_frame(fit.transition_matrix, transition, f"HMM K={n_states} transition")  # type: ignore[union-attr]
        _assert_frame(
            _computed_hmm_parameters(candidate),
            _persisted_hmm_parameters(n_states),
            f"HMM K={n_states} parameters",
        )

    comparison = build_model_comparison(markov_candidates, hmm_candidates)
    persisted_comparison = pd.read_csv(ROOT / "reports/tables/step3_model_comparison.csv")
    _assert_frame(comparison, persisted_comparison, "Step 3 model comparison")

    selection = select_preferred_model(comparison, markov_candidates, hmm_candidates)
    selected_meta = json.loads(
        (ROOT / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
    )
    for key in (
        "family",
        "n_states",
        "selection_reason",
        "markov_best_n_states",
        "hmm_best_n_states",
    ):
        if selection[key] != selected_meta[key]:
            raise AssertionError(f"selected-model metadata mismatch for {key}")
    return markov_candidates, hmm_candidates


def _verify_selected_analysis(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    meta_path = ROOT / "reports/generated/step3_selected_model.json"
    selected = json.loads(meta_path.read_text(encoding="utf-8"))
    step1_sha = hashlib.sha256((ROOT / "data/processed/step1_data.csv").read_bytes()).hexdigest()
    if selected["input_data_sha256"] != step1_sha:
        raise AssertionError("selected-model input SHA does not match Step 1 bytes")

    source_states = _states(selected["state_source"])
    selected_states = _states(selected["selected_states_path"])
    _assert_series(selected_states, source_states, "selected state path vs source path")
    if not selected_states.index.equals(data.index):
        raise AssertionError("selected state path does not align exactly with Step 1")

    statistics = compute_state_asset_statistics(data, selected_states)
    persisted_statistics = pd.read_csv(ROOT / "reports/tables/step3_state_asset_statistics.csv")
    _assert_frame(statistics, persisted_statistics, "Step 3 state/ETF statistics")

    allocation = build_state_allocation(statistics)
    persisted_allocation = pd.read_csv(ROOT / "reports/tables/step4_allocation_mapping.csv")
    _assert_frame(allocation, persisted_allocation, "Step 4 allocation mapping")
    return selected_states, allocation


def _verify_step5(data: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame) -> None:
    rotation = build_rotation_returns(data, states, allocation)
    comparison_index = pd.DatetimeIndex(rotation.index, name="Date")
    equal_weight = build_equal_weight_monthly_returns(data, comparison_index)
    spy = build_spy_buy_hold_returns(data, comparison_index)
    comparison = build_comparison(rotation, equal_weight, spy)
    persisted_daily = _indexed_csv("reports/tables/step5_daily_returns.csv")
    _assert_frame(comparison, persisted_daily, "Step 5 daily comparison returns")

    summary = build_performance_summary(comparison)
    persisted_summary = pd.read_csv(ROOT / "reports/tables/step5_performance_summary.csv")
    _assert_frame(summary, persisted_summary, "Step 5 performance summary")

    selected = json.loads(
        (ROOT / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
    )
    states_by_k = {
        n_states: _states(
            f"reports/tables/step2_{selected['family']}_{n_states}_states.csv"
        )
        for n_states in (2, 3)
    }
    sensitivity = build_state_count_sensitivity(data, selected["family"], states_by_k)
    sensitivity_path = ROOT / "reports/tables/step5_state_count_sensitivity.csv"
    if not sensitivity_path.is_file():
        raise AssertionError("missing canonical Step 5 sensitivity table")
    persisted_sensitivity = pd.read_csv(sensitivity_path)
    _assert_frame(sensitivity, persisted_sensitivity, "Step 5 state-count sensitivity")

    manifest_path = ROOT / "reports/generated/step5_manifest.json"
    if not manifest_path.is_file():
        raise AssertionError("missing canonical Step 5 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256((ROOT / "data/processed/step1_data.csv").read_bytes()).hexdigest()
    if manifest.get("input_data_sha256") != expected_sha:
        raise AssertionError("Step 5 manifest has a stale Step 1 SHA")
    for relative in manifest.get("tables", []) + manifest.get("figures", []):
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Step 5 manifest references missing artifact: {relative}")


def _verify_documentation_status() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stale_fragments = (
        "| Step 5 implementation | Not started |",
        "Step 5 remains unimplemented",
    )
    for fragment in stale_fragments:
        if fragment in readme:
            raise AssertionError(f"README contains stale implementation status: {fragment}")


def main() -> None:
    data = _step1()
    _verify_regime_models(data)
    selected_states, allocation = _verify_selected_analysis(data)
    _verify_step5(data, selected_states, allocation)
    _verify_documentation_status()
    print("Complete analysis consistency verification passed.")


if __name__ == "__main__":
    main()
