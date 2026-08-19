import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.markov_states import discretize_vix_change


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), name="Date")
    return pd.Series(values, index=index, name="VIX_change", dtype=float)


def test_two_state_quantile_and_right_boundary() -> None:
    series = _series([-2.0, -1.0, 0.0, 1.0, 2.0])
    states, thresholds = discretize_vix_change(series, 2)
    assert states.tolist() == [0, 0, 1, 1, 1]
    assert list(thresholds.columns) == ["state", "lower_bound", "upper_bound"]
    assert thresholds["state"].tolist() == [0, 1]
    assert thresholds.loc[0, "upper_bound"] == 0.0
    assert np.isneginf(thresholds.loc[0, "lower_bound"])
    assert np.isposinf(thresholds.loc[1, "upper_bound"])


def test_three_state_linear_quantiles() -> None:
    series = _series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    states, thresholds = discretize_vix_change(series, 3)
    expected_cuts = np.quantile(series.to_numpy(), [1 / 3, 2 / 3], method="linear")
    assert states.min() == 0 and states.max() == 2
    np.testing.assert_allclose(thresholds["upper_bound"].iloc[:2], expected_cuts)
    assert states.index.equals(series.index)
    assert states.name == "state"


def test_duplicate_cuts_and_invalid_inputs_fail() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        discretize_vix_change(_series([1.0, 1.0, 1.0, 1.0]), 3)
    with pytest.raises(ValueError, match="n_states"):
        discretize_vix_change(_series([0.0, 1.0]), 4)
    bad = _series([0.0, 1.0])
    bad.name = "wrong"
    with pytest.raises(ValueError, match="named"):
        discretize_vix_change(bad, 2)
    with pytest.raises(TypeError):
        discretize_vix_change(pd.DataFrame({"VIX_change": [1.0]}), 2)  # type: ignore[arg-type]
    nonfinite = _series([0.0, np.inf])
    with pytest.raises(ValueError, match="finite"):
        discretize_vix_change(nonfinite, 2)
