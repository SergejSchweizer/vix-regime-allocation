from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.selection import (
    build_validation_summary,
    candidate_grid,
    selected_configuration,
)


def _signals(n_states: int, family: str = "hmm") -> pd.DataFrame:
    decisions = pd.bdate_range("2015-01-02", periods=5)
    returns = pd.bdate_range("2015-01-05", periods=5)
    return pd.DataFrame(
        {
            "decision_date": decisions,
            "return_date": returns,
            "family": [family] * 5,
            "n_states": [n_states] * 5,
            "expected_TLT": [0.0] * 5,
            "expected_GLD": [0.001] * 5,
            "expected_SPY": [0.01] * 5,
        }
    )


def _assets(extra_date: str | None = None) -> pd.DataFrame:
    dates = list(pd.bdate_range("2015-01-05", periods=5))
    if extra_date is not None:
        dates.append(pd.Timestamp(extra_date))
    index = pd.DatetimeIndex(dates, name="Date")
    return pd.DataFrame(
        np.tile([0.001, 0.002, 0.003], (len(index), 1)),
        index=index,
        columns=["TLT", "GLD", "SPY"],
    )


def test_exact_hmm_only_grid_and_deterministic_tie_selection() -> None:
    grid = candidate_grid()
    assert len(grid) == 8
    assert {family for family, _, _ in grid} == {"hmm"}
    assert {k for _, k, _ in grid} == {2, 3}
    signals = {("hmm", k): _signals(k) for k in (2, 3)}

    summary = build_validation_summary(signals, _assets())

    assert len(summary) == 8
    assert summary["family"].unique().tolist() == ["hmm"]
    assert int(summary["selected"].sum()) == 1
    assert selected_configuration(summary) == ("hmm", 2, 0.0)


def test_selection_rejects_non_validation_dates() -> None:
    signals = {("hmm", k): _signals(k) for k in (2, 3)}
    signals[("hmm", 2)] = signals[("hmm", 2)].copy()
    signals[("hmm", 2)].loc[0, "return_date"] = pd.Timestamp("2021-01-04")
    with pytest.raises(ValueError, match="outside 2015-2020"):
        build_validation_summary(signals, _assets("2021-01-04"))


def test_markov_candidate_key_is_rejected() -> None:
    signals = {("hmm", 2): _signals(2), ("hmm", 3): _signals(3)}
    signals[("markov", 2)] = _signals(2, family="markov")
    with pytest.raises(ValueError, match="HMM K=2 and HMM K=3 exactly"):
        build_validation_summary(signals, _assets())


def test_markov_family_inside_signal_frame_is_rejected() -> None:
    signals = {("hmm", 2): _signals(2, family="markov"), ("hmm", 3): _signals(3)}
    with pytest.raises(ValueError, match="family='hmm' only"):
        build_validation_summary(signals, _assets())


def test_selected_configuration_rejects_markov_summary() -> None:
    summary = pd.DataFrame(
        [{"family": "markov", "n_states": 2, "switch_hurdle_bps": 0.0, "selected": True}]
    )
    with pytest.raises(ValueError, match="HMM candidates only"):
        selected_configuration(summary)
