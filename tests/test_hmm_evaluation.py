import pandas as pd
import pytest

from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.hmm_model import HMMFitResult


def _fit(index: pd.DatetimeIndex, n_states: int, log_likelihood: float) -> HMMFitResult:
    labels = [f"state_{state}" for state in range(n_states)]
    transition = pd.DataFrame(
        [[1.0 / n_states] * n_states for _ in range(n_states)],
        index=pd.Index(range(n_states), name="from_state"),
        columns=labels,
    )
    states = pd.Series([state % n_states for state in range(len(index))], index=index, name="state")
    return HMMFitResult(
        n_states=n_states,
        seed=42,
        converged=True,
        log_likelihood=log_likelihood,
        start_probabilities=tuple([1.0 / n_states] * n_states),
        transition_matrix=transition,
        means=pd.Series(range(n_states), index=pd.Index(range(n_states), name="state")),
        variances=pd.Series([1.0] * n_states, index=pd.Index(range(n_states), name="state")),
        states=states,
        probabilities=pd.DataFrame(
            [[1.0 / n_states] * n_states for _ in index], index=index, columns=labels
        ),
    )


@pytest.mark.parametrize(("n_states", "expected_parameters"), [(2, 7), (3, 14)])
def test_evaluation_delegates_once_and_uses_shared_ic(
    monkeypatch: pytest.MonkeyPatch, n_states: int, expected_parameters: int
) -> None:
    index = pd.date_range("2020-01-01", periods=6, name="Date")
    series = pd.Series([0.0, 1.0, 0.2, 1.1, -0.4, 0.7], index=index, name="VIX_change")
    fit = _fit(index, n_states, -5.0)
    calls = 0

    def fake_fit(vix_change: pd.Series, requested_states: int) -> HMMFitResult:
        nonlocal calls
        calls += 1
        assert vix_change is series
        assert requested_states == n_states
        return fit

    monkeypatch.setattr("vix_regime_allocation.hmm_evaluation.fit_gaussian_hmm", fake_fit)
    candidate = evaluate_hmm_candidate(series, n_states)
    assert calls == 1
    assert candidate["fit"] is fit
    assert candidate["family"] == "hmm"
    assert candidate["n_states"] == n_states
    assert candidate["n_parameters"] == expected_parameters
    assert candidate["n_observations"] == len(series)
    assert candidate["log_likelihood"] == -5.0
    assert candidate["aic"] == 2 * expected_parameters + 10.0
    assert candidate["converged"] is True
    assert float(candidate["bic"]) > 0.0
