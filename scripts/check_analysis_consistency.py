from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from vix_regime_allocation.allocation import build_state_allocation
from vix_regime_allocation.backtest_summary import build_four_portfolio_performance_summary
from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.model_selection import build_hmm_model_comparison, select_preferred_hmm
from vix_regime_allocation.sensitivity import build_hmm_state_count_sensitivity
from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.strategy_comparison import build_dual_method_comparison
from vix_regime_allocation.transform import OUTPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
STEP1 = ROOT / "data/processed/step1_data.csv"
RTOL = 1e-9
ATOL = 1e-11
TRADING_DAYS = 252


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
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise AssertionError("Step 1 Date index is invalid")
    values = data.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise AssertionError("Step 1 contains non-finite values")
    for asset in ("TLT", "GLD", "SPY"):
        expected = np.log(
            data[asset].iloc[1:].to_numpy(dtype=float)
            / data[asset].iloc[:-1].to_numpy(dtype=float)
        )
        np.testing.assert_allclose(
            data[f"{asset}_log_return"].iloc[1:].to_numpy(dtype=float),
            expected,
            rtol=RTOL,
            atol=ATOL,
        )
    np.testing.assert_allclose(
        data["VIX_change"].iloc[1:].to_numpy(dtype=float),
        data["VIX"].iloc[1:].to_numpy(dtype=float)
        - data["VIX"].iloc[:-1].to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
    )
    return data


def _computed_parameters(fit: HMMFitResult) -> pd.DataFrame:
    states = fit.states.to_numpy(dtype=int)
    counts = np.bincount(states, minlength=fit.n_states)
    return pd.DataFrame(
        {
            "state": np.arange(fit.n_states, dtype=int),
            "mean_vix_change": fit.means.to_numpy(dtype=float),
            "variance_vix_change": fit.variances.to_numpy(dtype=float),
            "start_probability": np.asarray(fit.start_probabilities, dtype=float),
            "viterbi_observations": counts,
            "viterbi_occupancy": counts.astype(float) / len(states),
            "posterior_mean_probability": fit.probabilities.mean(axis=0).to_numpy(dtype=float),
        }
    )


def _verify_hmms(data: pd.DataFrame) -> list[dict[str, object]]:
    vix_change = data["VIX_change"].rename("VIX_change")
    candidates: list[dict[str, object]] = []
    for n_states in (2, 3):
        candidate = evaluate_hmm_candidate(vix_change, n_states)
        candidates.append(candidate)
        fit = candidate.get("fit")
        if not isinstance(fit, HMMFitResult):
            raise AssertionError(f"HMM K={n_states} fit is missing")
        _assert_series(
            fit.states,
            _states(f"reports/tables/step2_hmm_{n_states}_states.csv"),
            f"HMM K={n_states} Viterbi states",
        )
        transition = pd.read_csv(
            ROOT / f"reports/tables/step2_hmm_{n_states}_transition.csv"
        ).set_index("from_state")
        transition.index.name = "from_state"
        _assert_frame(fit.transition_matrix, transition, f"HMM K={n_states} transition")
        parameters = pd.read_csv(ROOT / f"reports/tables/step2_hmm_{n_states}_parameters.csv")
        _assert_frame(_computed_parameters(fit), parameters, f"HMM K={n_states} parameters")
    return candidates


def _independent_weighted_returns(
    data: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame
) -> pd.Series:
    weights = allocation.set_index("state")[["TLT_weight", "GLD_weight", "SPY_weight"]]
    decisions = states.iloc[:-1].to_numpy(dtype=int)
    matrix = weights.loc[decisions].to_numpy(dtype=float)
    returns = np.expm1(
        data.loc[
            data.index[1:],
            ["TLT_log_return", "GLD_log_return", "SPY_log_return"],
        ].to_numpy(dtype=float)
    )
    return pd.Series(
        np.sum(matrix * returns, axis=1),
        index=pd.DatetimeIndex(data.index[1:], name="Date"),
        name="return",
    )


def _independent_metrics(returns: pd.Series) -> dict[str, float | int]:
    values = returns.to_numpy(dtype=float)
    n = len(values)
    wealth = np.cumprod(1.0 + values)
    terminal = float(wealth[-1])
    sample_std = float(np.std(values, ddof=1))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return {
        "cumulative_return": terminal - 1.0,
        "annualized_return": terminal ** (TRADING_DAYS / n) - 1.0,
        "annualized_volatility": sample_std * math.sqrt(TRADING_DAYS),
        "sharpe_ratio": float(np.mean(values)) / sample_std * math.sqrt(TRADING_DAYS),
        "max_drawdown": min(0.0, float(np.min(wealth / peaks - 1.0))),
        "observations": n,
    }


def _verify_new_canonical_outputs(
    data: pd.DataFrame,
    candidates: list[dict[str, object]],
    comparison: pd.DataFrame,
    selection: dict[str, object],
) -> None:
    comparison_path = ROOT / "reports/tables/step3_model_comparison.csv"
    keep_path = ROOT / "reports/tables/step4_allocation_100_keep.csv"
    spread_path = ROOT / "reports/tables/step4_allocation_60_40_spread.csv"
    if not (keep_path.is_file() and spread_path.is_file()):
        # Atomic source PRs precede PR-60, which commits the rebuilt canonical artifacts.
        return

    _assert_frame(comparison, pd.read_csv(comparison_path), "Step 3 HMM-only comparison")
    selected_states = selection["states"]
    if not isinstance(selected_states, pd.Series):
        raise AssertionError("HMM selection is missing states")
    selected_states = selected_states.astype("int64").rename("state")
    _assert_series(
        selected_states,
        _states("reports/tables/step3_selected_states.csv"),
        "selected HMM states",
    )

    selected_meta = json.loads(
        (ROOT / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
    )
    expected_sha = hashlib.sha256(STEP1.read_bytes()).hexdigest()
    for key, expected in {
        "family": "hmm",
        "n_states": int(selection["n_states"]),
        "state_source": str(selection["state_source"]),
        "input_data_sha256": expected_sha,
        "selected_states_path": "reports/tables/step3_selected_states.csv",
    }.items():
        if selected_meta.get(key) != expected:
            raise AssertionError(f"selected-model metadata mismatch for {key}")

    statistics = compute_state_asset_statistics(data, selected_states)
    _assert_frame(
        statistics,
        pd.read_csv(ROOT / "reports/tables/step3_state_asset_statistics.csv"),
        "Step 3 state/ETF statistics",
    )
    keep = build_state_allocation(statistics, "100_keep")
    spread = build_state_allocation(statistics, "60_40_spread")
    _assert_frame(keep, pd.read_csv(keep_path), "100% Keep allocation")
    _assert_frame(spread, pd.read_csv(spread_path), "60/40 Spread allocation")

    daily, _ = build_dual_method_comparison(data, selected_states, statistics)
    persisted_daily = _indexed_csv("reports/tables/step5_daily_returns.csv")
    _assert_frame(daily, persisted_daily, "Step 5 four-portfolio daily returns")
    for method, column, allocation in (
        ("100_keep", "hmm_100_keep", keep),
        ("60_40_spread", "hmm_60_40_spread", spread),
    ):
        independent = _independent_weighted_returns(data, selected_states, allocation)
        _assert_series(
            daily[column].rename("return"), independent, f"independent {method} lagged returns"
        )

    summary = build_four_portfolio_performance_summary(daily)
    persisted_summary = pd.read_csv(ROOT / "reports/tables/step5_performance_summary.csv")
    _assert_frame(summary, persisted_summary, "Step 5 performance summary")
    for row in persisted_summary.itertuples(index=False):
        portfolio = str(row.portfolio)
        direct = _independent_metrics(daily[portfolio])
        for key, expected in direct.items():
            actual = getattr(row, key)
            if key == "observations":
                if int(actual) != int(expected):
                    raise AssertionError(f"{portfolio} {key} mismatch")
            elif not math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
                raise AssertionError(f"{portfolio} {key} mismatch")

    states_by_k: dict[int, pd.Series] = {}
    for candidate in candidates:
        fit = candidate.get("fit")
        n_states = candidate.get("n_states")
        if not isinstance(fit, HMMFitResult) or not isinstance(n_states, int):
            raise AssertionError("invalid HMM candidate during sensitivity audit")
        states_by_k[n_states] = fit.states.astype("int64").rename("state")
    sensitivity = build_hmm_state_count_sensitivity(data, states_by_k)
    _assert_frame(
        sensitivity,
        pd.read_csv(ROOT / "reports/tables/step5_state_count_sensitivity.csv"),
        "HMM K-by-allocation sensitivity",
    )

    for relative in (
        "reports/generated/steps_2_4_manifest.json",
        "reports/generated/step5_manifest.json",
    ):
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        if payload.get("input_data_sha256") != expected_sha:
            raise AssertionError(f"{relative} has a stale Step 1 SHA")
        serialized = json.dumps(payload)
        if "markov" in serialized.lower():
            raise AssertionError(f"{relative} contains a non-HMM canonical artifact")
        for artifact in payload.get("tables", []) + payload.get("figures", []):
            path = ROOT / str(artifact)
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"{relative} references missing artifact: {artifact}")


def main() -> None:
    data = _step1()
    candidates = _verify_hmms(data)
    comparison = build_hmm_model_comparison(candidates)
    selection = select_preferred_hmm(comparison, candidates)
    if selection.get("family") != "hmm":
        raise AssertionError("HMM-only selection returned a non-HMM family")
    _verify_new_canonical_outputs(data, candidates, comparison, selection)
    print("HMM-only numerical analysis consistency checks passed.")


if __name__ == "__main__":
    main()
