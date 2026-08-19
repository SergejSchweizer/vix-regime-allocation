import math

import pytest

from vix_regime_allocation.information_criteria import (
    aic,
    bic,
    hmm_parameter_count,
    markov_parameter_count,
)


def test_exact_information_criteria_and_counts() -> None:
    assert aic(-10.0, 2) == 24.0
    assert bic(-10.0, 2, 100) == pytest.approx(2 * math.log(100) + 20)
    assert markov_parameter_count(2) == 2
    assert markov_parameter_count(3) == 6
    assert hmm_parameter_count(2) == 7
    assert hmm_parameter_count(3) == 14


def test_invalid_inputs_fail() -> None:
    for value in (math.inf, math.nan):
        with pytest.raises(ValueError):
            aic(value, 1)
    with pytest.raises(ValueError):
        aic(-1.0, -1)
    with pytest.raises(ValueError):
        aic(-1.0, True)
    with pytest.raises(ValueError):
        bic(-1.0, 1, 0)
    with pytest.raises(ValueError):
        bic(-1.0, 1, True)
    with pytest.raises(ValueError):
        markov_parameter_count(4)
    with pytest.raises(ValueError):
        hmm_parameter_count(1)
