from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.model_selection import (
    COMPARISON_COLUMNS,
    build_hmm_model_comparison,
    select_preferred_hmm,
)


def _index(periods: int = 40) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, name="Date")


def _fit(n_states: int = 2, periods: int = 40) -> HMMFitResult:
    index = _index(periods)
    labels = [f"state_{state}" for state in range(n_states)]
    states = np.resize(np.arange(n_states, dtype=int), periods)
    return HMMFitResult(
        n_states=n_states,
        seed=42,
        converged=True,
        log_likelihood=-20.0,
        start_probabilities=tuple([1.0 / n_states] * n_states),
        transition_matrix=pd.DataFrame(
            [[1.0 / n_states] * n_states for _ in range(n_states)],
            index=pd.Index(range(n_states), name="from_state"),
            columns=labels,
        ),
        means=pd.Series(
            np.arange(n_states, dtype=float), index=pd.Index(range(n_states), name="state")
        ),
        variances=pd.Series(np.ones(n_states), index=pd.Index(range(n_states), name="state")),
        states=pd.Series(states, index=index, name="state", dtype="int64"),
        probabilities=pd.DataFrame(
            [[1.0 / n_states] * n_states for _ in range(periods)], index=index, columns=labels
        ),
    )


def _candidate(
    n_states: int,
    bic: float,
    *,
    fit: HMMFitResult | None = None,
    converged: bool = True,
) -> dict[str, object]:
    actual_fit = fit if fit is not None else _fit(n_states)
    return {
        "family": "hmm",
        "n_states": n_states,
        "log_likelihood": actual_fit.log_likelihood,
        "n_parameters": n_states**2 + 2 * n_states - 1,
        "n_observations": len(actual_fit.states),
        "aic": 50.0 + n_states,
        "bic": bic,
        "converged": converged,
        "fit": actual_fit,
    }


def _candidates() -> list[dict[str, object]]:
    return [_candidate(2, 15.0), _candidate(3, 17.0)]


def test_build_hmm_model_comparison_has_exact_two_row_schema() -> None:
    comparison = build_hmm_model_comparison(_candidates())

    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert comparison[["family", "n_states"]].to_records(index=False).tolist() == [
        ("hmm", 2),
        ("hmm", 3),
    ]
    assert comparison["valid"].tolist() == [True, True]
    assert comparison["min_viterbi_occupancy"].min() >= 0.05


def test_select_preferred_hmm_uses_valid_candidate_bic_only() -> None:
    candidates = [_candidate(2, 15.0), _candidate(3, 14.0)]
    comparison = build_hmm_model_comparison(candidates)

    result = select_preferred_hmm(comparison, candidates)

    assert result["family"] == "hmm"
    assert result["n_states"] == 3
    assert result["state_source"] == "reports/tables/step2_hmm_3_states.csv"
    assert isinstance(result["states"], pd.Series)
    assert "BIC selected K=3" in str(result["selection_reason"])


def test_bic_tie_chooses_lower_k() -> None:
    candidates = [_candidate(2, 10.0 + 5e-13), _candidate(3, 10.0)]
    comparison = build_hmm_model_comparison(candidates)
    assert select_preferred_hmm(comparison, candidates)["n_states"] == 2


def test_invalid_low_bic_candidate_is_skipped() -> None:
    candidates = [_candidate(2, 50.0), _candidate(3, 1.0, converged=False)]
    comparison = build_hmm_model_comparison(candidates)
    assert comparison["valid"].tolist() == [True, False]
    assert select_preferred_hmm(comparison, candidates)["n_states"] == 2


def _with_bad_start(fit: HMMFitResult) -> HMMFitResult:
    return replace(fit, start_probabilities=tuple([0.8] * fit.n_states))


def _with_bad_transition(fit: HMMFitResult) -> HMMFitResult:
    transition = fit.transition_matrix.copy()
    transition.iloc[0, :] = 0.9
    return replace(fit, transition_matrix=transition)


def _with_bad_posterior(fit: HMMFitResult) -> HMMFitResult:
    posterior = fit.probabilities.copy()
    posterior.iloc[0, :] = 0.9
    return replace(fit, probabilities=posterior)


def _with_low_occupancy(fit: HMMFitResult) -> HMMFitResult:
    states = fit.states.copy()
    states[:] = 0
    states.iloc[-1] = fit.n_states - 1
    return replace(fit, states=states)


def _with_unordered_means(fit: HMMFitResult) -> HMMFitResult:
    means = fit.means.copy()
    means.iloc[0], means.iloc[1] = 2.0, 1.0
    return replace(fit, means=means)


@pytest.mark.parametrize(
    "transform",
    [
        lambda candidate: {**candidate, "converged": False},
        lambda candidate: {**candidate, "log_likelihood": float("nan")},
        lambda candidate: {**candidate, "fit": replace(candidate["fit"], converged=False)},
        lambda candidate: {**candidate, "fit": _with_unordered_means(candidate["fit"])},
        lambda candidate: {
            **candidate,
            "fit": replace(candidate["fit"], variances=pd.Series([1.0, 0.0])),
        },
        lambda candidate: {**candidate, "fit": _with_bad_start(candidate["fit"])},
        lambda candidate: {**candidate, "fit": _with_bad_transition(candidate["fit"])},
        lambda candidate: {**candidate, "fit": _with_bad_posterior(candidate["fit"])},
        lambda candidate: {**candidate, "fit": _with_low_occupancy(candidate["fit"])},
        lambda candidate: {**candidate, "fit": "not-a-fit"},
    ],
)
def test_each_validity_failure_is_recorded_without_alternate_family(transform: object) -> None:
    candidates = _candidates()
    assert callable(transform)
    candidates[0] = transform(candidates[0])  # type: ignore[operator]
    comparison = build_hmm_model_comparison(candidates)

    assert comparison.loc[0, "family"] == "hmm"
    assert bool(comparison.loc[0, "valid"]) is False
    result = select_preferred_hmm(comparison, candidates)
    assert result["family"] == "hmm"
    assert result["n_states"] == 3


def test_no_valid_hmm_fails_instead_of_falling_back() -> None:
    candidates = [_candidate(2, 1.0, converged=False), _candidate(3, 2.0, converged=False)]
    comparison = build_hmm_model_comparison(candidates)
    with pytest.raises(RuntimeError, match="No valid HMM candidate"):
        select_preferred_hmm(comparison, candidates)


def test_markov_candidate_is_rejected_by_hmm_public_api() -> None:
    candidates = _candidates()
    candidates[0] = {**candidates[0], "family": "markov"}
    with pytest.raises(ValueError, match="exactly 'hmm'"):
        build_hmm_model_comparison(candidates)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda candidates: candidates[:1],
        lambda candidates: [{**candidates[0], "family": "wrong"}, candidates[1]],
        lambda candidates: [{**candidates[0], "n_states": 3}, candidates[1]],
        lambda candidates: [{**candidates[0], "n_observations": 0}, candidates[1]],
        lambda candidates: [{**candidates[0], "converged": "yes"}, candidates[1]],
    ],
)
def test_malformed_candidate_sets_fail(mutator: object) -> None:
    candidates = _candidates()
    assert callable(mutator)
    bad = mutator(candidates)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        build_hmm_model_comparison(bad)


def test_malformed_comparison_fails() -> None:
    candidates = _candidates()
    comparison = build_hmm_model_comparison(candidates)
    with pytest.raises(TypeError, match="DataFrame"):
        select_preferred_hmm("bad", candidates)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns"):
        select_preferred_hmm(comparison.drop(columns="aic"), candidates)
    with pytest.raises(ValueError, match="exactly two"):
        select_preferred_hmm(comparison.iloc[:1], candidates)
    wrong_family = comparison.copy()
    wrong_family.loc[0, "family"] = "markov"
    with pytest.raises(ValueError, match="HMM only"):
        select_preferred_hmm(wrong_family, candidates)

# Synchronize PR quality gates against the current staged-rebuild base.
