from __future__ import annotations

import numpy as np
import pandas as pd

from vix_regime_allocation.predictive.markov_walkforward import build_markov_signals


def _data() -> pd.DataFrame:
    idx = pd.bdate_range("2020-12-15", periods=15, name="Date")
    x = np.linspace(-2.0, 2.0, len(idx))
    return pd.DataFrame(
        {
            "TLT_log_return": np.linspace(-0.01, 0.01, len(idx)),
            "GLD_log_return": np.linspace(0.005, -0.005, len(idx)),
            "SPY_log_return": np.linspace(-0.02, 0.02, len(idx)),
            "VIX_change": x,
        },
        index=idx,
    )


def test_markov_walkforward_is_causal_and_prefix_invariant() -> None:
    data = _data()
    decisions = data.index[8:13]
    result = build_markov_signals(data, decisions, 2)
    assert len(result) == len(decisions)
    assert (result["training_end"] < result["decision_date"]).all()
    assert result["return_date"].tolist() == data.index[9:14].tolist()
    assert np.allclose(result[["p_state_0", "p_state_1"]].sum(axis=1), 1.0)

    future_date = data.index[-1] + pd.offsets.BDay(1)
    future = pd.DataFrame(
        {
            "TLT_log_return": [0.5],
            "GLD_log_return": [-0.4],
            "SPY_log_return": [0.3],
            "VIX_change": [99.0],
        },
        index=pd.DatetimeIndex([future_date], name="Date"),
    )
    extended = pd.concat([data, future])
    repeated = build_markov_signals(extended, decisions, 2)
    pd.testing.assert_frame_equal(result, repeated)
