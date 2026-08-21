from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.sensitivity as module
from vix_regime_allocation.performance import PERFORMANCE_KEYS
from vix_regime_allocation.sensitivity import (
    HMM_SENSITIVITY_COLUMNS,
    METHOD_ORDER,
    build_hmm_state_count_sensitivity,
)
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _data() -> pd.DataFrame:
    index = pd.date_range("2026-01-02", periods=6, freq="B", name="Date")
    frame = pd.DataFrame(index=index)
    for asset, level in (("TLT", 100.0), ("GLD", 200.0), ("SPY", 300.0)):
        frame[asset] = level + np.arange(6)
        frame[f"{asset}_log_return"] = [0.00, 0.01, -0.01, 0.02, -0.02, 0.01]
    frame["VIX"] = [20.0, 21.0, 19.0, 22.0, 18.0, 23.0]
    frame["VIX_change"] = [0.0, 1.0, -2.0, 3.0, -4.0, 5.0]
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def _states(data: pd.DataFrame) -> dict[int, pd.Series]:
    return {
        2: pd.Series([0, 0, 0, 1, 1, 1], index=data.index, name="state", dtype=int),
        3: pd.Series([0, 0, 1, 1, 2, 2], index=data.index, name="state", dtype=int),
    }


def test_hmm_sensitivity_exact_four_rows_and_common_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    states_by_k = _states(data)
    calls: list[tuple[str, int, str]] = []

    def fake_stats(_: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
        k = int(states.nunique())
        calls.append(("stats", k, ""))
        return pd.DataFrame({"k": [k]})

    def fake_allocation(stats: pd.DataFrame, method: str = "100_keep") -> pd.DataFrame:
        k = int(stats.loc[0, "k"])
        calls.append(("allocation", k, method))
        return pd.DataFrame({"k": [k], "method": [method]})

    def fake_rotation(
        _: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame
    ) -> pd.DataFrame:
        k = int(states.nunique())
        method = str(allocation.loc[0, "method"])
        calls.append(("rotation", k, method))
        start = 1 if k == 2 and method == "100_keep" else 2
        index = data.index[start:]
        value = 0.001 * k + (0.0001 if method == "60_40_spread" else 0.0)
        return pd.DataFrame({"regime_rotation_return": np.full(len(index), value)}, index=index)

    metric_calls: list[tuple[str, int, str]] = []

    def fake_metrics(series: pd.Series) -> dict[str, float | int]:
        metric_calls.append((str(series.index[0].date()), len(series), str(series.name)))
        n = float(len(metric_calls))
        return {
            "cumulative_return": n,
            "annualized_return": n + 0.1,
            "annualized_volatility": n + 0.2,
            "sharpe_ratio": n + 0.3,
            "max_drawdown": -n / 10.0,
            "observations": len(series),
        }

    monkeypatch.setattr(module, "compute_state_asset_statistics", fake_stats)
    monkeypatch.setattr(module, "build_state_allocation", fake_allocation)
    monkeypatch.setattr(module, "build_rotation_returns", fake_rotation)
    monkeypatch.setattr(module, "performance_metrics", fake_metrics)

    result = build_hmm_state_count_sensitivity(data, states_by_k)

    assert tuple(result.columns) == HMM_SENSITIVITY_COLUMNS
    assert result[["family", "n_states", "method"]].to_records(index=False).tolist() == [
        ("hmm", 2, "100_keep"),
        ("hmm", 2, "60_40_spread"),
        ("hmm", 3, "100_keep"),
        ("hmm", 3, "60_40_spread"),
    ]
    assert result["observations"].tolist() == [4, 4, 4, 4]
    assert calls == [
        ("stats", 2, ""),
        ("allocation", 2, "100_keep"),
        ("rotation", 2, "100_keep"),
        ("allocation", 2, "60_40_spread"),
        ("rotation", 2, "60_40_spread"),
        ("stats", 3, ""),
        ("allocation", 3, "100_keep"),
        ("rotation", 3, "100_keep"),
        ("allocation", 3, "60_40_spread"),
        ("rotation", 3, "60_40_spread"),
    ]
    assert [row[:2] for row in metric_calls] == [
        (str(data.index[2].date()), 4),
        (str(data.index[2].date()), 4),
        (str(data.index[2].date()), 4),
        (str(data.index[2].date()), 4),
    ]
    assert {name for _, _, name in metric_calls} == {
        "hmm_k2_100_keep",
        "hmm_k2_60_40_spread",
        "hmm_k3_100_keep",
        "hmm_k3_60_40_spread",
    }


def test_hmm_sensitivity_uses_both_methods_for_each_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    allocations: list[str] = []
    original = module.build_state_allocation

    def spy(stats: pd.DataFrame, method: str = "100_keep") -> pd.DataFrame:
        allocations.append(method)
        return original(stats, method)

    monkeypatch.setattr(module, "build_state_allocation", spy)
    result = build_hmm_state_count_sensitivity(data, _states(data))
    assert allocations == ["100_keep", "60_40_spread", "100_keep", "60_40_spread"]
    assert result["method"].tolist() == list(METHOD_ORDER) * 2
    assert result.loc[result["n_states"] == 3, "observations"].nunique() == 1


def test_hmm_sensitivity_rejects_unexpected_metric_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()

    def malformed(_: pd.Series) -> dict[str, float | int]:
        return {key: 1.0 for key in PERFORMANCE_KEYS if key != "observations"}

    monkeypatch.setattr(module, "performance_metrics", malformed)
    with pytest.raises(ValueError, match="unexpected metric schema"):
        build_hmm_state_count_sensitivity(data, _states(data))


def test_hmm_sensitivity_has_no_family_argument() -> None:
    data = _data()
    with pytest.raises(TypeError):
        build_hmm_state_count_sensitivity(data, "markov", _states(data))  # type: ignore[call-arg]


def test_hmm_sensitivity_rejects_wrong_state_mapping_contract() -> None:
    data = _data()
    with pytest.raises(TypeError):
        build_hmm_state_count_sensitivity(data, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_hmm_state_count_sensitivity(data, {2: _states(data)[2]})


@pytest.mark.parametrize("case", ["name", "index", "dtype", "labels", "count"])
def test_hmm_sensitivity_rejects_invalid_states(case: str) -> None:
    data = _data()
    states_by_k = _states(data)
    states = states_by_k[3].copy()
    if case == "name":
        states = states.rename("regime")
    elif case == "index":
        states = states.iloc[::-1]
    elif case == "dtype":
        states = states.astype(float)
    elif case == "labels":
        states = pd.Series([0, 0, 1, 1, 3, 3], index=data.index, name="state", dtype=int)
    else:
        states = pd.Series([0, 0, 0, 1, 1, 2], index=data.index, name="state", dtype=int)
    states_by_k[3] = states
    with pytest.raises(ValueError):
        build_hmm_state_count_sensitivity(data, states_by_k)


@pytest.mark.parametrize(
    "case",
    [
        "type",
        "schema",
        "index_type",
        "index_name",
        "duplicate",
        "short",
        "nonnumeric",
        "nonfinite",
    ],
)
def test_hmm_sensitivity_rejects_invalid_data(case: str) -> None:
    data: object = _data().copy()
    if case == "type":
        data = []
    elif case == "schema":
        data = _data().drop(columns=["VIX_change"])
    elif case == "index_type":
        frame = _data().copy()
        frame.index = pd.Index(range(len(frame)), name="Date")
        data = frame
    elif case == "index_name":
        frame = _data().copy()
        frame.index = frame.index.rename("date")
        data = frame
    elif case == "duplicate":
        frame = _data().copy()
        frame.index = pd.DatetimeIndex(
            ["2026-01-02", "2026-01-02", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
            name="Date",
        )
        data = frame
    elif case == "short":
        data = _data().iloc[:3]
    elif case == "nonnumeric":
        frame = _data().copy()
        frame["TLT"] = "bad"
        data = frame
    else:
        frame = _data().copy()
        frame.iloc[0, 0] = np.inf
        data = frame

    error = TypeError if case == "type" else ValueError
    with pytest.raises(error):
        build_hmm_state_count_sensitivity(data, {})  # type: ignore[arg-type]
