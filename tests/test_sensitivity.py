from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.sensitivity as module
from vix_regime_allocation.performance import PERFORMANCE_KEYS
from vix_regime_allocation.sensitivity import SENSITIVITY_COLUMNS, build_state_count_sensitivity
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


def test_sensitivity_delegates_all_shared_steps_and_uses_common_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    states_by_k = _states(data)
    calls: list[tuple[str, int]] = []

    def fake_stats(_: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
        k = int(states.nunique())
        calls.append(("stats", k))
        return pd.DataFrame({"k": [k]})

    def fake_allocation(stats: pd.DataFrame) -> pd.DataFrame:
        k = int(stats.loc[0, "k"])
        calls.append(("allocation", k))
        return pd.DataFrame({"k": [k]})

    def fake_rotation(_: pd.DataFrame, states: pd.Series, allocation: pd.DataFrame) -> pd.DataFrame:
        k = int(states.nunique())
        assert int(allocation.loc[0, "k"]) == k
        calls.append(("rotation", k))
        index = data.index[1:] if k == 2 else data.index[2:]
        return pd.DataFrame({"regime_rotation_return": np.full(len(index), 0.001 * k)}, index=index)

    metric_calls: list[tuple[str, int]] = []

    def fake_metrics(series: pd.Series) -> dict[str, float | int]:
        k = 2 if series.name == "regime_rotation_k2" else 3
        metric_calls.append((str(series.index[0].date()), len(series)))
        return {
            "cumulative_return": float(k),
            "annualized_return": float(k) + 0.1,
            "annualized_volatility": float(k) + 0.2,
            "sharpe_ratio": float(k) + 0.3,
            "max_drawdown": -float(k) / 10.0,
            "observations": len(series),
        }

    monkeypatch.setattr(module, "compute_state_asset_statistics", fake_stats)
    monkeypatch.setattr(module, "build_state_allocation", fake_allocation)
    monkeypatch.setattr(module, "build_rotation_returns", fake_rotation)
    monkeypatch.setattr(module, "performance_metrics", fake_metrics)

    result = build_state_count_sensitivity(data, "markov", states_by_k)

    assert calls == [
        ("stats", 2),
        ("allocation", 2),
        ("rotation", 2),
        ("stats", 3),
        ("allocation", 3),
        ("rotation", 3),
    ]
    assert metric_calls == [(str(data.index[2].date()), 4), (str(data.index[2].date()), 4)]
    assert tuple(result.columns) == SENSITIVITY_COLUMNS
    assert result["family"].tolist() == ["markov", "markov"]
    assert result["n_states"].tolist() == [2, 3]
    assert result["observations"].tolist() == [4, 4]
    assert result["cumulative_return"].tolist() == [2.0, 3.0]


def test_sensitivity_rejects_unexpected_metric_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data()

    def malformed(_: pd.Series) -> dict[str, float | int]:
        return {key: 1.0 for key in PERFORMANCE_KEYS if key != "observations"}

    monkeypatch.setattr(module, "performance_metrics", malformed)
    with pytest.raises(ValueError, match="unexpected metric schema"):
        build_state_count_sensitivity(data, "markov", _states(data))


@pytest.mark.parametrize("family", ["", "other", "MARKOV"])
def test_sensitivity_rejects_invalid_family(family: str) -> None:
    data = _data()
    with pytest.raises(ValueError):
        build_state_count_sensitivity(data, family, _states(data))


def test_sensitivity_rejects_wrong_state_mapping_contract() -> None:
    data = _data()
    with pytest.raises(TypeError):
        build_state_count_sensitivity(data, "markov", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_state_count_sensitivity(data, "markov", {2: _states(data)[2]})


@pytest.mark.parametrize("case", ["name", "index", "dtype", "labels", "count"])
def test_sensitivity_rejects_invalid_states(case: str) -> None:
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
        build_state_count_sensitivity(data, "markov", states_by_k)


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
def test_sensitivity_rejects_invalid_data(case: str) -> None:
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
            [
                "2026-01-02",
                "2026-01-02",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
            ],
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
        build_state_count_sensitivity(data, "markov", {})  # type: ignore[arg-type]
