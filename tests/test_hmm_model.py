from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.hmm_model as hmm_module
from vix_regime_allocation.hmm_model import fit_gaussian_hmm


def _series() -> pd.Series:
    rng = np.random.default_rng(123)
    values = np.concatenate((rng.normal(-2.0, 0.25, 60), rng.normal(2.0, 0.3, 60)))
    index = pd.date_range("2020-01-01", periods=len(values), name="Date")
    return pd.Series(values, index=index, name="VIX_change")


@pytest.mark.parametrize("n_states", [2, 3])
def test_fit_shapes_order_and_probabilities(n_states: int) -> None:
    series = _series()
    result = fit_gaussian_hmm(series, n_states)
    assert result.n_states == n_states
    assert result.seed in (42, 43, 44, 45, 46)
    assert result.converged is True
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.diff(result.means.to_numpy()) >= 0.0)
    assert np.all(result.variances.to_numpy() > 0.0)
    assert result.states.index.equals(series.index)
    assert set(result.states.unique()) <= set(range(n_states))
    assert list(result.probabilities.columns) == [f"state_{i}" for i in range(n_states)]
    np.testing.assert_allclose(result.probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert sum(result.start_probabilities) == pytest.approx(1.0)


def test_configured_settings_and_all_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            seen.append(kwargs)
            self.monitor_ = SimpleNamespace(converged=True)

        def fit(self, observations: np.ndarray) -> "FakeModel":
            assert observations.shape == (4, 1)
            return self

        def score(self, observations: np.ndarray) -> float:
            del observations
            return float(seen[-1]["random_state"])

    monkeypatch.setattr(hmm_module, "GaussianHMM", FakeModel)
    _, seed, score = hmm_module._select_restart(np.array([[0.0], [1.0], [0.2], [1.1]]), 2)
    assert [item["random_state"] for item in seen] == [42, 43, 44, 45, 46]
    assert seed == 46
    assert score == 46.0


def test_failed_restart_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        monitor_ = SimpleNamespace(converged=True)

    attempted: list[int] = []

    def fake_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[FakeModel, float]:
        del observations, n_states
        attempted.append(seed)
        if seed == 43:
            raise ValueError("singular covariance")
        return FakeModel(), float(seed)

    monkeypatch.setattr(hmm_module, "_fit_restart", fake_restart)
    _, seed, score = hmm_module._select_restart(np.zeros((4, 1)), 2)
    assert attempted == [42, 43, 44, 45, 46]
    assert seed == 46
    assert score == 46.0


def test_likelihood_tie_uses_smallest_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        monitor_ = SimpleNamespace(converged=True)

    def fake_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[FakeModel, float]:
        del observations, n_states
        return FakeModel(), 10.0 if seed in (42, 43) else 9.0

    monkeypatch.setattr(hmm_module, "_fit_restart", fake_restart)
    _, seed, score = hmm_module._select_restart(np.zeros((4, 1)), 2)
    assert seed == 42
    assert score == 10.0


def test_no_converged_restart_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        monitor_ = SimpleNamespace(converged=False)

    def fake_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[FakeModel, float]:
        del observations, n_states, seed
        return FakeModel(), -1.0

    monkeypatch.setattr(hmm_module, "_fit_restart", fake_restart)
    with pytest.raises(RuntimeError, match="No configured HMM restart"):
        hmm_module._select_restart(np.zeros((4, 1)), 2)


def test_all_restarts_failing_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[object, float]:
        del observations, n_states, seed
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(hmm_module, "_fit_restart", fake_restart)
    with pytest.raises(RuntimeError, match="No configured HMM restart"):
        hmm_module._select_restart(np.zeros((4, 1)), 2)


def test_invalid_series_and_state_count_fail() -> None:
    series = _series()
    with pytest.raises(ValueError, match="n_states"):
        fit_gaussian_hmm(series, 4)
    with pytest.raises(ValueError, match="named"):
        fit_gaussian_hmm(series.rename("wrong"), 2)
    with pytest.raises(ValueError, match="enough"):
        fit_gaussian_hmm(series.iloc[:2], 3)
