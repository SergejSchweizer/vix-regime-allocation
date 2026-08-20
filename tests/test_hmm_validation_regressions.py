from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.hmm_model as hmm_module


def test_one_failed_restart_does_not_abort_remaining_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        monitor_ = SimpleNamespace(converged=True)

    def fake_restart(observations: np.ndarray, n_states: int, seed: int) -> tuple[FakeModel, float]:
        del observations, n_states
        if seed == 42:
            raise ValueError("singular initialization")
        return FakeModel(), float(seed)

    monkeypatch.setattr(hmm_module, "_fit_restart", fake_restart)
    _, seed, score = hmm_module._select_restart(np.zeros((8, 1)), 2)

    assert seed == 46
    assert score == 46.0


def test_relabel_rejects_nonfinite_state_means() -> None:
    class FakeModel:
        n_components = 2
        means_ = np.array([[np.nan], [1.0]])
        covars_ = np.array([1.0, 1.0])

    index = pd.date_range("2026-01-01", periods=4, name="Date")
    series = pd.Series([0.0, 0.1, -0.1, 0.2], index=index, name="VIX_change")

    with pytest.raises(ValueError, match="means must be finite"):
        hmm_module._relabel_result(
            FakeModel(),  # type: ignore[arg-type]
            42,
            -1.0,
            series.to_numpy(dtype=float).reshape(-1, 1),
            series,
        )


def test_relabel_rejects_wrong_posterior_shape() -> None:
    class FakeModel:
        n_components = 2
        means_ = np.array([[-1.0], [1.0]])
        covars_ = np.array([1.0, 1.0])
        startprob_ = np.array([0.5, 0.5])
        transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])

        def predict(self, observations: np.ndarray) -> np.ndarray:
            return np.zeros(len(observations), dtype=int)

        def predict_proba(self, observations: np.ndarray) -> np.ndarray:
            return np.ones((len(observations) - 1, 2), dtype=float) / 2.0

    index = pd.date_range("2026-01-01", periods=4, name="Date")
    series = pd.Series([0.0, 0.1, -0.1, 0.2], index=index, name="VIX_change")

    with pytest.raises(ValueError, match="invalid shape"):
        hmm_module._relabel_result(
            FakeModel(),  # type: ignore[arg-type]
            42,
            -1.0,
            series.to_numpy(dtype=float).reshape(-1, 1),
            series,
        )
