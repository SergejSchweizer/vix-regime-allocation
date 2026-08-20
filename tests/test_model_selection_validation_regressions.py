from dataclasses import replace

import numpy as np
import pandas as pd

from vix_regime_allocation.hmm_model import HMMFitResult
from vix_regime_allocation.model_selection import build_model_comparison, select_preferred_model


def _index(periods: int = 20) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=periods, name="Date")


def _fit() -> HMMFitResult:
    index = _index()
    return HMMFitResult(
        n_states=2,
        seed=42,
        converged=True,
        log_likelihood=-20.0,
        start_probabilities=(0.5, 0.5),
        transition_matrix=pd.DataFrame(
            [[0.9, 0.1], [0.2, 0.8]],
            index=pd.Index([0, 1], name="from_state"),
            columns=["state_0", "state_1"],
        ),
        means=pd.Series([-1.0, 1.0], index=pd.Index([0, 1], name="state")),
        variances=pd.Series([1.0, 2.0], index=pd.Index([0, 1], name="state")),
        states=pd.Series([0, 1] * 10, index=index, name="state", dtype="int64"),
        probabilities=pd.DataFrame(
            [[0.6, 0.4], [0.4, 0.6]] * 10,
            index=index,
            columns=["state_0", "state_1"],
        ),
    )


def _markov(n_states: int, bic: float) -> dict[str, object]:
    return {
        "family": "markov",
        "n_states": n_states,
        "log_likelihood": -10.0,
        "n_parameters": n_states * (n_states - 1),
        "n_observations": 19,
        "aic": 25.0,
        "bic": bic,
        "converged": True,
        "states": pd.Series(
            [i % n_states for i in range(20)],
            index=_index(),
            name="state",
            dtype="int64",
        ),
    }


def _hmm(n_states: int, bic: float, fit: HMMFitResult) -> dict[str, object]:
    return {
        "family": "hmm",
        "n_states": n_states,
        "log_likelihood": fit.log_likelihood,
        "n_parameters": n_states**2 + 2 * n_states - 1,
        "n_observations": len(fit.states),
        "aic": 45.0,
        "bic": bic,
        "converged": True,
        "fit": fit,
    }


def test_nonfinite_hmm_mean_forces_markov_fallback() -> None:
    base = _fit()
    bad = replace(
        base,
        means=pd.Series([np.nan, 1.0], index=pd.Index([0, 1], name="state")),
    )
    markov = [_markov(2, 10.0), _markov(3, 11.0)]
    hmm = [_hmm(2, 5.0, bad), _hmm(3, 7.0, replace(base, n_states=3))]
    comparison = build_model_comparison(markov, hmm)

    result = select_preferred_model(comparison, markov, hmm)

    assert result["family"] == "markov"
    assert "state means are not finite" in str(result["selection_reason"])


def test_misaligned_hmm_posterior_dates_force_markov_fallback() -> None:
    base = _fit()
    shifted = base.probabilities.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    bad = replace(base, probabilities=shifted)
    markov = [_markov(2, 10.0), _markov(3, 11.0)]
    hmm = [_hmm(2, 5.0, bad), _hmm(3, 7.0, replace(base, n_states=3))]
    comparison = build_model_comparison(markov, hmm)

    result = select_preferred_model(comparison, markov, hmm)

    assert result["family"] == "markov"
    assert "misaligned" in str(result["selection_reason"])
