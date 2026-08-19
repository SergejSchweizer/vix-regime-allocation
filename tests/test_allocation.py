from __future__ import annotations

import pandas as pd
import pytest

from vix_regime_allocation.allocation import ALLOCATION_COLUMNS, build_state_allocation
from vix_regime_allocation.state_statistics import STATISTICS_COLUMNS


def _statistics(means_by_state: list[dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state, means in enumerate(means_by_state):
        for asset in ("TLT", "GLD", "SPY"):
            rows.append(
                {
                    "state": state,
                    "asset": asset,
                    "mean_log_return": means[asset],
                    "std_log_return": 0.01 + state * 0.001,
                    "observations": 10 + state,
                }
            )
    return pd.DataFrame(rows, columns=list(STATISTICS_COLUMNS))


def test_build_state_allocation_selects_all_possible_winners() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
            {"TLT": 0.001, "GLD": 0.004, "SPY": 0.002},
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.005},
        ]
    )

    result = build_state_allocation(statistics)

    assert tuple(result.columns) == ALLOCATION_COLUMNS
    assert result["state"].tolist() == [0, 1, 2]
    assert result["selected_asset"].tolist() == ["TLT", "GLD", "SPY"]
    assert result["selection_mean_log_return"].tolist() == [0.003, 0.004, 0.005]
    assert result[["TLT_weight", "GLD_weight", "SPY_weight"]].values.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert result[["TLT_weight", "GLD_weight", "SPY_weight"]].sum(axis=1).tolist() == [
        1.0,
        1.0,
        1.0,
    ]


def test_build_state_allocation_uses_fixed_two_way_tie_priority() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.003, "GLD": 0.003, "SPY": 0.001},
            {"TLT": 0.001, "GLD": 0.004, "SPY": 0.004},
        ]
    )

    result = build_state_allocation(statistics)

    assert result["selected_asset"].tolist() == ["TLT", "GLD"]
    assert result["selection_mean_log_return"].tolist() == [0.003, 0.004]


def test_build_state_allocation_uses_tlt_for_three_way_tie() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.002, "GLD": 0.002, "SPY": 0.002},
            {"TLT": -0.001, "GLD": -0.002, "SPY": -0.003},
        ]
    )

    result = build_state_allocation(statistics)

    assert result.loc[0, "selected_asset"] == "TLT"
    assert result.loc[0, "TLT_weight"] == 1.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns=["std_log_return"]),
        lambda frame: frame.assign(asset=frame["asset"].replace({"SPY": "QQQ"})),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.loc[~((frame["state"] == 0) & (frame["asset"] == "SPY"))].copy(),
        lambda frame: frame.assign(mean_log_return=float("nan")),
        lambda frame: frame.assign(std_log_return=-0.1),
        lambda frame: frame.assign(observations=1),
    ],
)
def test_build_state_allocation_rejects_malformed_statistics(mutator) -> None:  # type: ignore[no-untyped-def]
    statistics = _statistics(
        [
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.003},
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
        ]
    )

    with pytest.raises(ValueError):
        build_state_allocation(mutator(statistics))


def test_build_state_allocation_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        build_state_allocation([])  # type: ignore[arg-type]
