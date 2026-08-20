from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.policy import (
    apply_transaction_cost,
    choose_asset,
    one_hot_weights,
    turnover,
)


def test_choice_ties_hurdle_and_turnover() -> None:
    tied = pd.Series([0.01, 0.01, 0.01], index=["TLT", "GLD", "SPY"])
    assert choose_asset(tied, None, 0.0) == "TLT"

    expected = pd.Series([0.0100, 0.0107, 0.0090], index=["TLT", "GLD", "SPY"])
    assert choose_asset(expected, "TLT", 10.0) == "TLT"
    assert choose_asset(expected, "TLT", 5.0) == "GLD"

    zero = np.zeros(3)
    tlt = one_hot_weights("TLT")
    spy = one_hot_weights("SPY")
    assert turnover(zero, tlt) == 0.5
    assert turnover(tlt, spy) == 1.0


def test_transaction_cost_and_validation() -> None:
    assert apply_transaction_cost(0.01, 1.0, 5.0) == pytest.approx(0.0095)
    with pytest.raises(ValueError):
        choose_asset(pd.Series([1.0, 2.0], index=["TLT", "GLD"]), None, 0.0)
    with pytest.raises(ValueError):
        one_hot_weights("CASH")
    with pytest.raises(ValueError):
        apply_transaction_cost(-1.0, 0.0)
