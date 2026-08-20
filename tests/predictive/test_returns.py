from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.predictive.returns import asset_simple_returns, buy_and_hold_returns


def _data() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]), name="Date")
    return pd.DataFrame(
        {
            "TLT_log_return": [np.log(1.01), np.log(0.99)],
            "GLD_log_return": [np.log(1.02), 0.0],
            "SPY_log_return": [0.0, np.log(1.03)],
        },
        index=idx,
    )


def test_asset_simple_returns_and_buy_hold() -> None:
    result = asset_simple_returns(_data())
    np.testing.assert_allclose(result.to_numpy(), [[0.01, 0.02, 0.0], [-0.01, 0.0, 0.03]])
    assert list(result.columns) == ["TLT", "GLD", "SPY"]
    copied = buy_and_hold_returns(result)
    pd.testing.assert_frame_equal(copied, result)
    assert copied is not result


def test_return_validation_errors() -> None:
    with pytest.raises(TypeError):
        asset_simple_returns("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        asset_simple_returns(_data().drop(columns=["SPY_log_return"]))
    bad = _data()
    bad.loc[bad.index[0], "TLT_log_return"] = np.nan
    with pytest.raises(ValueError):
        asset_simple_returns(bad)
    simple = asset_simple_returns(_data())
    with pytest.raises(ValueError):
        buy_and_hold_returns(simple.rename(columns={"TLT": "X"}))
