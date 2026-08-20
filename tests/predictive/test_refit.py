from __future__ import annotations

import pandas as pd
import pytest

from vix_regime_allocation.predictive.refit import build_refit_schedule


def test_refit_schedule_uses_previous_observed_row() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05", "2021-02-01", "2021-02-02"]),
        name="Date",
    )
    result = build_refit_schedule(index, pd.Timestamp("2021-01-04"))
    assert result["decision_date"].tolist() == pd.to_datetime(
        ["2021-01-04", "2021-01-05", "2021-02-01"]
    ).tolist()
    assert result["refit"].tolist() == [True, False, True]
    assert result["training_end"].tolist() == pd.to_datetime(
        ["2020-12-31", "2020-12-31", "2021-01-05"]
    ).tolist()
    assert (result["training_end"] < result["decision_date"]).all()


def test_refit_schedule_errors() -> None:
    idx = pd.date_range("2021-01-01", periods=3, name="Date")
    with pytest.raises(ValueError):
        build_refit_schedule(idx, pd.Timestamp("2020-01-01"))
    with pytest.raises(ValueError):
        build_refit_schedule(idx, idx[0])
