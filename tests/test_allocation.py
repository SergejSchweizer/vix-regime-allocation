from __future__ import annotations

import itertools

import pandas as pd
import pytest

from vix_regime_allocation.allocation import (
    ALLOCATION_COLUMNS,
    METHOD_ALLOCATION_COLUMNS,
    build_state_allocation,
)
from vix_regime_allocation.state_statistics import STATISTICS_COLUMNS

ASSETS = ("TLT", "GLD", "SPY")


def _statistics(means_by_state: list[dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state, means in enumerate(means_by_state):
        for asset in ASSETS:
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


def test_both_methods_share_exact_ranking_and_fixed_weights() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
            {"TLT": 0.001, "GLD": 0.004, "SPY": 0.002},
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.005},
        ]
    )
    keep = build_state_allocation(statistics, "100_keep")
    spread = build_state_allocation(statistics, "60_40_spread")

    assert tuple(keep.columns) == METHOD_ALLOCATION_COLUMNS
    assert tuple(spread.columns) == METHOD_ALLOCATION_COLUMNS
    assert keep["rank_1_asset"].tolist() == ["TLT", "GLD", "SPY"]
    assert spread["rank_1_asset"].tolist() == keep["rank_1_asset"].tolist()
    assert spread["rank_2_asset"].tolist() == keep["rank_2_asset"].tolist()
    assert keep[["TLT_weight", "GLD_weight", "SPY_weight"]].values.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert spread[["TLT_weight", "GLD_weight", "SPY_weight"]].values.tolist() == [
        [0.6, 0.4, 0.0],
        [0.4, 0.6, 0.0],
        [0.0, 0.4, 0.6],
    ]
    for frame in (keep, spread):
        assert frame[["TLT_weight", "GLD_weight", "SPY_weight"]].sum(axis=1).tolist() == [
            1.0,
            1.0,
            1.0,
        ]


@pytest.mark.parametrize("ranking", list(itertools.permutations(ASSETS)))
def test_all_rank_permutations_are_deterministic(ranking: tuple[str, str, str]) -> None:
    means = {asset: float(3 - position) for position, asset in enumerate(ranking)}
    statistics = _statistics([means, {"TLT": 3.0, "GLD": 2.0, "SPY": 1.0}])
    result = build_state_allocation(statistics, "60_40_spread")
    assert result.loc[0, "rank_1_asset"] == ranking[0]
    assert result.loc[0, "rank_2_asset"] == ranking[1]
    assert result.loc[0, f"{ranking[0]}_weight"] == pytest.approx(0.6)
    assert result.loc[0, f"{ranking[1]}_weight"] == pytest.approx(0.4)
    assert result.loc[0, f"{ranking[2]}_weight"] == pytest.approx(0.0)


def test_fixed_tie_priority_applies_to_rank_one_and_rank_two() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.003, "GLD": 0.003, "SPY": 0.001},
            {"TLT": 0.001, "GLD": 0.004, "SPY": 0.004},
            {"TLT": 0.002, "GLD": 0.002, "SPY": 0.002},
        ]
    )
    result = build_state_allocation(statistics, "60_40_spread")
    assert result[["rank_1_asset", "rank_2_asset"]].values.tolist() == [
        ["TLT", "GLD"],
        ["GLD", "SPY"],
        ["TLT", "GLD"],
    ]


def test_omitted_method_preserves_temporary_legacy_100_keep_schema() -> None:
    statistics = _statistics(
        [
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.003},
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
        ]
    )
    result = build_state_allocation(statistics)
    assert tuple(result.columns) == ALLOCATION_COLUMNS
    assert result["selected_asset"].tolist() == ["SPY", "TLT"]
    assert result[["TLT_weight", "GLD_weight", "SPY_weight"]].values.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize("method", ["", "70_30", "markov", "100%"])
def test_unsupported_method_fails(method: str) -> None:
    statistics = _statistics(
        [
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.003},
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
        ]
    )
    with pytest.raises(ValueError, match="method must be one of"):
        build_state_allocation(statistics, method)


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
        build_state_allocation(mutator(statistics), "100_keep")


def test_build_state_allocation_rejects_non_dataframe_and_non_string_method() -> None:
    with pytest.raises(TypeError):
        build_state_allocation([], "100_keep")  # type: ignore[arg-type]
    statistics = _statistics(
        [
            {"TLT": 0.001, "GLD": 0.002, "SPY": 0.003},
            {"TLT": 0.003, "GLD": 0.002, "SPY": 0.001},
        ]
    )
    with pytest.raises(TypeError, match="method must be a string"):
        build_state_allocation(statistics, 1)  # type: ignore[arg-type]

# Synchronize PR quality gates against the current staged-rebuild base.
