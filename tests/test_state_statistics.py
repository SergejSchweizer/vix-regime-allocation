import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.state_statistics import (
    ASSET_ORDER,
    STATISTICS_COLUMNS,
    compute_state_asset_statistics,
)
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _data() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=6, name="Date")
    frame = pd.DataFrame(
        {
            "TLT": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "GLD": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
            "SPY": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
            "VIX": [15.0, 16.0, 14.0, 17.0, 18.0, 16.0],
            "TLT_log_return": [0.01, 0.03, -0.02, 0.05, 0.00, 0.04],
            "GLD_log_return": [0.02, -0.01, 0.04, 0.06, -0.03, 0.01],
            "SPY_log_return": [-0.01, 0.02, 0.03, -0.02, 0.05, 0.00],
            "VIX_change": [1.0, 1.0, -2.0, 3.0, 1.0, -2.0],
        },
        index=index,
    )
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def _states() -> pd.Series:
    return pd.Series([0, 0, 1, 1, 0, 1], index=_data().index, name="state", dtype="int64")


def test_statistics_exact_schema_order_values_and_ddof_one() -> None:
    data = _data()
    states = _states()

    result = compute_state_asset_statistics(data, states)

    assert tuple(result.columns) == STATISTICS_COLUMNS
    assert result[["state", "asset"]].to_records(index=False).tolist() == [
        (0, "TLT"),
        (0, "GLD"),
        (0, "SPY"),
        (1, "TLT"),
        (1, "GLD"),
        (1, "SPY"),
    ]
    assert tuple(result.loc[result["state"] == 0, "asset"]) == ASSET_ORDER

    state_zero_tlt = result.loc[(result["state"] == 0) & (result["asset"] == "TLT")].iloc[0]
    expected_zero = np.array([0.01, 0.03, 0.00])
    assert state_zero_tlt["mean_log_return"] == pytest.approx(expected_zero.mean())
    assert state_zero_tlt["std_log_return"] == pytest.approx(expected_zero.std(ddof=1))
    assert state_zero_tlt["observations"] == 3

    state_one_gld = result.loc[(result["state"] == 1) & (result["asset"] == "GLD")].iloc[0]
    expected_one = np.array([0.04, 0.06, 0.01])
    assert state_one_gld["mean_log_return"] == pytest.approx(expected_one.mean())
    assert state_one_gld["std_log_return"] == pytest.approx(expected_one.std(ddof=1))
    assert state_one_gld["observations"] == 3


def test_three_state_order_and_counts() -> None:
    data = _data()
    states = pd.Series([0, 0, 1, 1, 2, 2], index=data.index, name="state", dtype="int64")
    result = compute_state_asset_statistics(data, states)

    assert len(result) == 9
    assert result.groupby("state", sort=True)["observations"].first().tolist() == [2, 2, 2]
    assert result["state"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize(
    "bad_data",
    [
        lambda frame: frame.rename(columns={"VIX_change": "wrong"}),
        lambda frame: frame.rename_axis("wrong"),
        lambda frame: frame.sort_index(ascending=False),
        lambda frame: pd.concat([frame.iloc[[0]], frame]),
        lambda frame: frame.assign(TLT_log_return=np.nan),
        lambda frame: frame.assign(SPY_log_return=np.inf),
        lambda frame: frame.assign(GLD_log_return="not-numeric"),
    ],
)
def test_invalid_data_contract_fails(bad_data: object) -> None:
    frame = _data()
    transform = bad_data
    assert callable(transform)
    invalid = transform(frame)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        compute_state_asset_statistics(invalid, _states())


def test_non_dataframe_data_fails() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        compute_state_asset_statistics("bad", _states())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "states",
    [
        pd.Series([0, 0, 1, 1, 0, 1], index=_data().index, name="wrong", dtype="int64"),
        pd.Series([0, 0, 1, 1, 0, 1], index=_data().index[::-1], name="state", dtype="int64"),
        pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], index=_data().index, name="state"),
        pd.Series([0, 0, 2, 2, 0, 2], index=_data().index, name="state", dtype="int64"),
        pd.Series([0, 0, 1, 1, 2, 3], index=_data().index, name="state", dtype="int64"),
        pd.Series([0, 0, 0, 0, 0, 1], index=_data().index, name="state", dtype="int64"),
    ],
)
def test_invalid_state_contract_fails(states: pd.Series) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_state_asset_statistics(_data(), states)


def test_non_series_states_fail() -> None:
    with pytest.raises(TypeError, match="Series"):
        compute_state_asset_statistics(_data(), [0, 0, 1, 1, 0, 1])  # type: ignore[arg-type]
