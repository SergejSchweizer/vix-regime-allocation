import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.markov_chain import estimate_transition_matrix, stationary_distribution


def _states(values: list[int]) -> pd.Series:
    return pd.Series(values, name="state", dtype="int64")


def test_transition_matrix_and_stationary_distribution_manual() -> None:
    states = _states([0, 0, 1, 0, 1, 1, 0])
    transition = estimate_transition_matrix(states, 2)
    expected = np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
    np.testing.assert_allclose(transition.to_numpy(), expected)
    stationary = stationary_distribution(transition)
    np.testing.assert_allclose(stationary.to_numpy(), [0.5, 0.5], atol=1e-10)
    np.testing.assert_allclose(
        stationary.to_numpy() @ transition.to_numpy(), stationary.to_numpy()
    )


def test_zero_outgoing_and_invalid_states_fail() -> None:
    with pytest.raises(ValueError, match="outgoing"):
        estimate_transition_matrix(_states([0, 0, 0, 1]), 2)
    with pytest.raises(ValueError, match="outside"):
        estimate_transition_matrix(_states([0, 2, 0]), 2)
    bad = pd.Series([0.0, 1.0], name="state")
    with pytest.raises(ValueError, match="integer"):
        estimate_transition_matrix(bad, 2)


def test_nonunique_and_malformed_transition_fail() -> None:
    nonunique = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index([0, 1], name="from_state"),
        columns=["state_0", "state_1"],
    )
    with pytest.raises(ValueError, match="unique"):
        stationary_distribution(nonunique)
    malformed = nonunique.copy()
    malformed.columns = ["x", "y"]
    with pytest.raises(ValueError, match="columns"):
        stationary_distribution(malformed)
    nonstochastic = nonunique.copy()
    nonstochastic.iloc[0] = [0.2, 0.2]
    with pytest.raises(ValueError, match="sum"):
        stationary_distribution(nonstochastic)
