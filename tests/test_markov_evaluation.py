import math

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.markov_evaluation as evaluation
from vix_regime_allocation.markov_evaluation import evaluate_markov_candidate


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), name="Date")
    return pd.Series(values, index=index, name="VIX_change", dtype=float)


@pytest.mark.parametrize(("n_states", "expected_parameters"), [(2, 2), (3, 6)])
def test_candidate_math_and_exact_keys(n_states: int, expected_parameters: int) -> None:
    series = _series([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5])
    candidate = evaluate_markov_candidate(series, n_states)
    expected_keys = {
        "family",
        "n_states",
        "log_likelihood",
        "n_parameters",
        "n_observations",
        "aic",
        "bic",
        "converged",
        "thresholds",
        "transition",
        "stationary",
        "states",
    }
    assert set(candidate) == expected_keys
    assert candidate["family"] == "markov"
    assert candidate["n_states"] == n_states
    assert candidate["n_parameters"] == expected_parameters
    assert candidate["n_observations"] == len(series) - 1
    assert candidate["converged"] is True
    log_likelihood = float(candidate["log_likelihood"])
    assert float(candidate["aic"]) == pytest.approx(2 * expected_parameters - 2 * log_likelihood)
    expected_bic = expected_parameters * math.log(len(series) - 1) - 2 * log_likelihood
    assert float(candidate["bic"]) == pytest.approx(expected_bic)


def test_conditional_likelihood_matches_manual_transitions() -> None:
    states = pd.Series([0, 0, 1, 0], name="state", dtype="int64")
    transition = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0]],
        index=pd.Index([0, 1], name="from_state"),
        columns=["state_0", "state_1"],
    )
    result = evaluation._conditional_log_likelihood(states, transition)
    assert result == pytest.approx(math.log(0.5) + math.log(0.5) + math.log(1.0))


def test_impossible_transition_fails() -> None:
    states = pd.Series([0, 1], name="state", dtype="int64")
    transition = pd.DataFrame(
        [[1.0, 0.0], [0.5, 0.5]],
        index=pd.Index([0, 1], name="from_state"),
        columns=["state_0", "state_1"],
    )
    with pytest.raises(ValueError, match="positive"):
        evaluation._conditional_log_likelihood(states, transition)


def test_candidate_delegates_to_shared_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    series = _series([-1.0, 1.0, -0.5, 0.5])
    states = pd.Series([0, 1, 0, 1], index=series.index, name="state", dtype="int64")
    thresholds = pd.DataFrame(
        {"state": [0, 1], "lower_bound": [-np.inf, 0.0], "upper_bound": [0.0, np.inf]}
    )
    transition = pd.DataFrame(
        [[0.25, 0.75], [0.5, 0.5]],
        index=pd.Index([0, 1], name="from_state"),
        columns=["state_0", "state_1"],
    )
    stationary = pd.Series(
        [0.4, 0.6], index=pd.Index([0, 1], name="state"), name="stationary_probability"
    )
    calls: list[str] = []

    def fake_discretize(vix_change: pd.Series, n_states: int) -> tuple[pd.Series, pd.DataFrame]:
        assert vix_change is series and n_states == 2
        calls.append("states")
        return states, thresholds

    def fake_transition(input_states: pd.Series, n_states: int) -> pd.DataFrame:
        assert input_states is states and n_states == 2
        calls.append("transition")
        return transition

    def fake_stationary(input_transition: pd.DataFrame) -> pd.Series:
        assert input_transition is transition
        calls.append("stationary")
        return stationary

    monkeypatch.setattr(evaluation, "discretize_vix_change", fake_discretize)
    monkeypatch.setattr(evaluation, "estimate_transition_matrix", fake_transition)
    monkeypatch.setattr(evaluation, "stationary_distribution", fake_stationary)
    candidate = evaluation.evaluate_markov_candidate(series, 2)
    assert calls == ["states", "transition", "stationary"]
    assert candidate["states"] is states
    assert candidate["thresholds"] is thresholds
    assert candidate["transition"] is transition
    assert candidate["stationary"] is stationary
