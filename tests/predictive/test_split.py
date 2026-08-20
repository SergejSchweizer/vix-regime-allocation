from __future__ import annotations

import pandas as pd
import pytest

from vix_regime_allocation.predictive.config import (
    ASSET_ORDER,
    ONE_WAY_COST_BPS,
    SUPPORTED_STATE_COUNTS,
    SWITCH_HURDLES_BPS,
)
from vix_regime_allocation.predictive.split import is_monthly_refit, split_periods


def _frame() -> pd.DataFrame:
    index = pd.to_datetime(
        ["2014-12-30", "2014-12-31", "2015-01-02", "2020-12-31", "2021-01-04", "2021-01-05"]
    )
    return pd.DataFrame({"x": range(len(index))}, index=pd.DatetimeIndex(index, name="Date"))


def test_fixed_configuration() -> None:
    assert ASSET_ORDER == ("TLT", "GLD", "SPY")
    assert SUPPORTED_STATE_COUNTS == (2, 3)
    assert SWITCH_HURDLES_BPS == (0.0, 5.0, 10.0, 20.0)
    assert ONE_WAY_COST_BPS == 5.0


def test_split_periods_resolves_fixed_calendar_boundaries() -> None:
    periods = split_periods(_frame())
    assert periods.initial_history.tolist() == pd.to_datetime(["2014-12-30", "2014-12-31"]).tolist()
    assert periods.validation.tolist() == pd.to_datetime(["2015-01-02", "2020-12-31"]).tolist()
    assert periods.test.tolist() == pd.to_datetime(["2021-01-04", "2021-01-05"]).tolist()


def test_split_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        split_periods("bad")  # type: ignore[arg-type]
    bad = _frame().copy()
    bad.index = bad.index[::-1]
    with pytest.raises(ValueError):
        split_periods(bad)
    with pytest.raises(ValueError):
        split_periods(_frame().loc[:"2020-12-31"])


def test_monthly_refit_semantics() -> None:
    assert is_monthly_refit(None, pd.Timestamp("2021-01-04"))
    assert not is_monthly_refit(pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05"))
    assert is_monthly_refit(pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-01"))
    with pytest.raises(ValueError):
        is_monthly_refit(pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-05"))
