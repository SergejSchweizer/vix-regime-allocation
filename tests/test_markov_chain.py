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
    np.testing.assert_allclose(stationary.to_numpy() @ transition.to_numpy(), stationary.to_numpy())


def test_three_state_transition_is_row_stochastic_and_stationary() -> None:
    states = _states([0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 0])
    transition = estimate_transition_matrix(states, 3)
    np.testing.assert_allclose(transition.sum(axis=1).to_numpy(), 1.0)
    stationary = stationary_distribution(transition)
    assert stationary.name == "stationary_probability"
    assert stationary.index.name == "state"
    assert (stationary >= 0.0).all()
    assert stationary.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(stationary.to_numpy() @ transition.to_numpy(), stationary.to_numpy())


def test_state_validation_failures_are_explicit() -> None:
    with pytest.raises(ValueError, match="n_states"):
        estimate_transition_matrix(_states([0, 1]), 4)
    with pytest.raises(TypeError, match="Series"):
        estimate_transition_matrix([0, 1], 2)  # type: ignore[arg-type]
    wrong_name = _states([0, 1]).rename("wrong")
    with pytest.raises(ValueError, match="named"):
        estimate_transition_matrix(wrong_name, 2)
    with pytest.raises(ValueError, match="At least two"):
        estimate_transition_matrix(_states([0]), 2)
    bad_dtype = pd.Series([0.0, 1.0], name="state")
    with pytest.raises(ValueError, match="integer"):
        estimate_transition_matrix(bad_dtype, 2)
    with pytest.raises(ValueError, match="outside"):
        estimate_transition_matrix(_states([0, 2, 0]), 2)
    with pytest.raises(ValueError, match="outgoing"):
        estimate_transition_matrix(_states([0, 0, 0, 1]), 2)


def test_transition_validation_failures_are_explicit() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        stationary_distribution(np.eye(2))  # type: ignore[arg-type]

    one_state = pd.DataFrame([[1.0]], index=pd.Index([0], name="from_state"), columns=["state_0"])
    with pytest.raises(ValueError, match="two or three"):
        stationary_distribution(one_state)

    identity = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index([0, 1], name="from_state"),
        columns=["state_0", "state_1"],
    )
    malformed_columns = identity.copy()
    malformed_columns.columns = ["x", "y"]
    with pytest.raises(ValueError, match="columns"):
        stationary_distribution(malformed_columns)

    malformed_index = identity.copy()
    malformed_index.index = pd.Index([1, 0], name="from_state")
    with pytest.raises(ValueError, match="index"):
        stationary_distribution(malformed_index)

    unnamed_index = identity.copy()
    unnamed_index.index.name = None
    with pytest.raises(ValueError, match="index"):
        stationary_distribution(unnamed_index)

    negative = identity.copy()
    negative.iloc[0] = [1.1, -0.1]
    with pytest.raises(ValueError, match="non-negative"):
        stationary_distribution(negative)

    nonfinite = identity.copy()
    nonfinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        stationary_distribution(nonfinite)

    nonstochastic = identity.copy()
    nonstochastic.iloc[0] = [0.2, 0.2]
    with pytest.raises(ValueError, match="sum"):
        stationary_distribution(nonstochastic)

    with pytest.raises(ValueError, match="unique"):
        stationary_distribution(identity)
