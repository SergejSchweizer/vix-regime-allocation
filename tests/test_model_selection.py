from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.model_selection import (
    COMPARISON_COLUMNS,
    build_model_comparison,
    select_preferred_model,
)


def _index(periods: int = 20) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, name="Date")


def _fit(n_states: int = 2, periods: int = 20) -> HMMFitResult:
    index = _index(periods)
    labels = [f"state_{state}" for state in range(n_states)]
    states = [state % n_states for state in range(periods)]
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
        variances=pd.Series(
            np.ones(n_states), index=pd.Index(range(n_states), name="state")
        ),
        states=pd.Series(states, index=index, name="state", dtype="int64"),
        probabilities=pd.DataFrame(
            [[1.0 / n_states] * n_states for _ in range(periods)],
            index=index,
            columns=labels,
        ),
    )


def _markov_candidate(n_states: int, bic: float) -> dict[str, object]:
    states = pd.Series(
        [state % n_states for state in range(20)],
        index=_index(),
        name="state",
        dtype="int64",
    )
    return {
        "family": "markov",
        "n_states": n_states,
        "log_likelihood": -10.0 - n_states,
        "n_parameters": n_states * (n_states - 1),
        "n_observations": 19,
        "aic": 30.0 + n_states,
        "bic": bic,
        "converged": True,
        "states": states,
    }


def _hmm_candidate(
    n_states: int, bic: float, *, fit: HMMFitResult | None = None, converged: bool = True
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


def _candidates() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    markov = [_markov_candidate(2, 20.0), _markov_candidate(3, 18.0)]
    hmm = [_hmm_candidate(2, 15.0), _hmm_candidate(3, 17.0)]
    return markov, hmm


def test_build_model_comparison_has_exact_order_schema_and_scope() -> None:
    markov, hmm = _candidates()
    comparison = build_model_comparison(markov, hmm)

    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert list(zip(comparison["family"], comparison["n_states"], strict=True)) == [
        ("markov", 2),
        ("markov", 3),
        ("hmm", 2),
        ("hmm", 3),
    ]
    assert comparison["criterion_scope"].tolist() == ["within_family_only"] * 4
    assert comparison.loc[0, "bic"] == 20.0
    assert comparison.loc[2, "aic"] == 52.0


def test_selects_valid_hmm_using_within_family_bic_only() -> None:
    markov = [_markov_candidate(2, -1000.0), _markov_candidate(3, -900.0)]
    hmm = [_hmm_candidate(2, 15.0), _hmm_candidate(3, 17.0)]
    comparison = build_model_comparison(markov, hmm)

    result = select_preferred_model(comparison, markov, hmm)

    assert result["family"] == "hmm"
    assert result["n_states"] == 2
    assert result["markov_best_n_states"] == 2
    assert result["hmm_best_n_states"] == 2
    assert result["state_source"] == "reports/tables/step2_hmm_2_states.csv"
    assert isinstance(result["states"], pd.Series)
    assert "all fixed HMM validity diagnostics passed" in str(result["selection_reason"])


def test_bic_tie_within_each_family_chooses_lower_k() -> None:
    markov = [_markov_candidate(2, 10.0), _markov_candidate(3, 10.0 + 5e-13)]
    hmm = [_hmm_candidate(2, 8.0 + 5e-13), _hmm_candidate(3, 8.0)]
    comparison = build_model_comparison(markov, hmm)

    result = select_preferred_model(comparison, markov, hmm)

    assert result["markov_best_n_states"] == 2
    assert result["hmm_best_n_states"] == 2
    assert result["n_states"] == 2


def _with_bad_start(fit: HMMFitResult) -> HMMFitResult:
    return replace(fit, start_probabilities=(0.8, 0.8))


def _with_bad_transition(fit: HMMFitResult) -> HMMFitResult:
    transition = fit.transition_matrix.copy()
    transition.iloc[0, :] = [0.9, 0.9]
    return replace(fit, transition_matrix=transition)


def _with_bad_posterior(fit: HMMFitResult) -> HMMFitResult:
    posterior = fit.probabilities.copy()
    posterior.iloc[0, :] = [0.9, 0.9]
    return replace(fit, probabilities=posterior)


def _with_low_occupancy(fit: HMMFitResult) -> HMMFitResult:
    states = pd.Series([0] * len(fit.states), index=fit.states.index, name="state", dtype="int64")
    return replace(fit, states=states)


@pytest.mark.parametrize(
    ("candidate_transform", "reason"),
    [
        (lambda candidate: {**candidate, "converged": False}, "candidate did not converge"),
        (
            lambda candidate: {**candidate, "fit": replace(candidate["fit"], converged=False)},
            "selected HMM fit did not converge",
        ),
        (
            lambda candidate: {
                **candidate,
                "fit": replace(candidate["fit"], variances=pd.Series([1.0, 0.0])),
            },
            "state variances",
        ),
        (
            lambda candidate: {**candidate, "fit": _with_bad_start(candidate["fit"])},
            "initial-state probabilities",
        ),
        (
            lambda candidate: {**candidate, "fit": _with_bad_transition(candidate["fit"])},
            "transition probabilities",
        ),
        (
            lambda candidate: {**candidate, "fit": _with_bad_posterior(candidate["fit"])},
            "posterior probabilities",
        ),
        (
            lambda candidate: {**candidate, "fit": _with_low_occupancy(candidate["fit"])},
            "occupancy below 5%",
        ),
        (lambda candidate: {**candidate, "fit": "not-a-fit"}, "fit result is missing"),
    ],
)
def test_each_hmm_invalidity_falls_back_to_markov(
    candidate_transform: object, reason: str
) -> None:
    markov = [_markov_candidate(2, 10.0), _markov_candidate(3, 11.0)]
    hmm = [_hmm_candidate(2, 5.0), _hmm_candidate(3, 7.0)]
    transform = candidate_transform
    assert callable(transform)
    hmm[0] = transform(hmm[0])  # type: ignore[operator]
    comparison = build_model_comparison(markov, hmm)

    result = select_preferred_model(comparison, markov, hmm)

    assert result["family"] == "markov"
    assert result["n_states"] == 2
    assert result["state_source"] == "reports/tables/step2_markov_2_states.csv"
    assert reason in str(result["selection_reason"])


def test_fallback_requires_selected_markov_state_series() -> None:
    markov = [_markov_candidate(2, 10.0), _markov_candidate(3, 11.0)]
    markov[0].pop("states")
    hmm = [_hmm_candidate(2, 5.0, converged=False), _hmm_candidate(3, 7.0)]
    comparison = build_model_comparison(markov, hmm)

    with pytest.raises(ValueError, match="canonical state Series"):
        select_preferred_model(comparison, markov, hmm)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda markov, hmm: (markov[:1], hmm),
        lambda markov, hmm: ([{**markov[0], "family": "wrong"}, markov[1]], hmm),
        lambda markov, hmm: ([{**markov[0], "n_states": 3}, markov[1]], hmm),
        lambda markov, hmm: ([{**markov[0], "bic": float("nan")}, markov[1]], hmm),
        lambda markov, hmm: ([{**markov[0], "n_observations": 0}, markov[1]], hmm),
        lambda markov, hmm: (markov, [{**hmm[0], "converged": "yes"}, hmm[1]]),
    ],
)
def test_malformed_candidate_sets_fail(mutator: object) -> None:
    markov, hmm = _candidates()
    transform = mutator
    assert callable(transform)
    bad_markov, bad_hmm = transform(markov, hmm)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        build_model_comparison(bad_markov, bad_hmm)


def test_malformed_comparison_and_scope_fail() -> None:
    markov, hmm = _candidates()
    comparison = build_model_comparison(markov, hmm)

    with pytest.raises(TypeError, match="DataFrame"):
        select_preferred_model("not-a-frame", markov, hmm)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="columns"):
        select_preferred_model(comparison.drop(columns="aic"), markov, hmm)

    bad_scope = comparison.copy()
    bad_scope.loc[0, "criterion_scope"] = "cross_family"
    with pytest.raises(ValueError, match="within model family"):
        select_preferred_model(bad_scope, markov, hmm)

    bad_rows = comparison.iloc[:3].copy()
    with pytest.raises(ValueError, match="exactly four"):
        select_preferred_model(bad_rows, markov, hmm)
