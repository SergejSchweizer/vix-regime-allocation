from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.state_returns import (
    expected_asset_returns,
    hard_state_asset_means,
    soft_state_asset_means,
)


def _returns() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, name="Date")
    return pd.DataFrame(
        {"TLT": [0.0, 0.02, 0.01, 0.03], "GLD": [0.01, 0.03, 0.02, 0.04], "SPY": [0.02, 0.04, 0.03, 0.05]},
        index=idx,
    )


def test_hard_soft_and_expected_returns() -> None:
    returns = _returns()
    states = pd.Series([0, 0, 1, 1], index=returns.index, name="state")
    hard = hard_state_asset_means(returns, states, 2)
    np.testing.assert_allclose(hard.loc[0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(hard.loc[1], [0.02, 0.03, 0.04])

    probs = pd.DataFrame(
        {"state_0": [1.0, 1.0, 0.0, 0.0], "state_1": [0.0, 0.0, 1.0, 1.0]},
        index=returns.index,
    )
    soft = soft_state_asset_means(returns, probs)
    pd.testing.assert_frame_equal(hard, soft)
    expected = expected_asset_returns(np.array([0.25, 0.75]), hard)
    np.testing.assert_allclose(expected.to_numpy(), [0.0175, 0.0275, 0.0375])


def test_state_return_validation() -> None:
    returns = _returns()
    states = pd.Series([0, 0, 0, 0], index=returns.index)
    with pytest.raises(ValueError):
        hard_state_asset_means(returns, states, 2)
    probs = pd.DataFrame({"state_0": [1.0] * 4, "state_1": [0.0] * 4}, index=returns.index)
    with pytest.raises(ValueError):
        soft_state_asset_means(returns, probs)
