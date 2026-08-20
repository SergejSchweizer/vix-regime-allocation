from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vix_regime_allocation.predictive.plots import (
    comparison_return_frame,
    plot_cumulative_performance,
    plot_regime_forecast_probabilities,
    probability_columns,
)


def _data() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]),
        name="Date",
    )
    simple = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03],
            [-0.01, 0.01, -0.02],
            [0.02, 0.00, 0.01],
        ]
    )
    return pd.DataFrame(
        {
            "TLT": [100.0] * 4,
            "GLD": [100.0] * 4,
            "SPY": [100.0] * 4,
            "VIX": [20.0] * 4,
            "TLT_log_return": np.log1p(simple[:, 0]),
            "GLD_log_return": np.log1p(simple[:, 1]),
            "SPY_log_return": np.log1p(simple[:, 2]),
            "VIX_change": [0.0] * 4,
        },
        index=idx,
    )


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
            "return_date": pd.to_datetime(["2021-01-05", "2021-01-06", "2021-01-07"]),
            "gross_return": [0.03, 0.01, 0.02],
            "net_return": [0.02975, 0.0095, 0.0195],
            "p_state_0": [0.8, 0.6, 0.4],
            "p_state_1": [0.2, 0.4, 0.6],
        }
    )


def test_comparison_contains_every_required_series() -> None:
    frame = comparison_return_frame(_data(), _daily())
    assert list(frame.columns) == [
        "Predictive gross",
        "Predictive net",
        "TLT",
        "GLD",
        "SPY",
        "Equal weight",
    ]
    assert probability_columns(_daily()) == ["p_state_0", "p_state_1"]


def test_predictive_plots_write_non_empty_pngs(tmp_path: Path) -> None:
    cumulative = plot_cumulative_performance(_data(), _daily(), tmp_path / "cumulative.png")
    probabilities = plot_regime_forecast_probabilities(_daily(), tmp_path / "probabilities.png")
    assert cumulative.stat().st_size > 0
    assert probabilities.stat().st_size > 0
