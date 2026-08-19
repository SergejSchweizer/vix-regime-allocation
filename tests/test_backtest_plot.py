from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.backtest_plot as module
from vix_regime_allocation.backtest_plot import COMPARISON_COLUMNS, plot_cumulative_performance


def _comparison() -> pd.DataFrame:
    index = pd.date_range("2026-01-02", periods=3, freq="B", name="Date")
    return pd.DataFrame(
        {
            "regime_rotation": [0.01, 0.02, -0.01],
            "equal_weight_monthly": [0.00, 0.01, 0.02],
            "spy_buy_hold": [-0.01, 0.03, 0.00],
        },
        index=index,
    )


def test_plot_delegates_once_per_portfolio_and_drawdown_includes_initial_wealth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comparison = _comparison()
    calls: list[str] = []

    def fake_wealth(series: pd.Series) -> pd.Series:
        calls.append(str(series.name))
        return (1.0 + series).cumprod().rename("wealth")

    monkeypatch.setattr(module, "cumulative_wealth", fake_wealth)
    output = tmp_path / "nested" / "figure.png"
    before = set(plt.get_fignums())
    plot_cumulative_performance(comparison, output)

    assert calls == list(COMPARISON_COLUMNS)
    assert output.exists() and output.stat().st_size > 0
    assert set(plt.get_fignums()) == before

    spy_wealth = (1.0 + comparison["spy_buy_hold"]).cumprod()
    expected_first_drawdown = spy_wealth.iloc[0] / 1.0 - 1.0
    assert expected_first_drawdown == pytest.approx(-0.01)


def test_saved_figure_has_cumulative_and_drawdown_panels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    original_savefig = plt.Figure.savefig

    def recording_savefig(self: plt.Figure, *args: object, **kwargs: object) -> None:
        cumulative_axis, drawdown_axis = self.axes
        observed["title"] = cumulative_axis.get_title()
        observed["cum_ylabel"] = cumulative_axis.get_ylabel()
        observed["dd_ylabel"] = drawdown_axis.get_ylabel()
        observed["xlabel"] = drawdown_axis.get_xlabel()
        observed["cum_lines"] = [
            line.get_label() for line in cumulative_axis.lines if not str(line.get_label()).startswith("_")
        ]
        observed["dd_lines"] = [
            line.get_label() for line in drawdown_axis.lines if not str(line.get_label()).startswith("_")
        ]
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", recording_savefig)
    plot_cumulative_performance(_comparison(), tmp_path / "figure.png")
    assert observed["title"] == "Step 5 Performance Comparison"
    assert observed["cum_ylabel"] == "Cumulative return"
    assert observed["dd_ylabel"] == "Drawdown"
    assert observed["xlabel"] == "Date"
    assert observed["cum_lines"] == [
        "Regime rotation",
        "Equal weight (monthly reset)",
        "SPY buy and hold",
    ]
    assert observed["dd_lines"] == observed["cum_lines"]


@pytest.mark.parametrize(
    "case",
    [
        "not_dataframe",
        "columns",
        "index_type",
        "index_name",
        "empty",
        "duplicate",
        "nonnumeric",
        "nonfinite",
        "invalid_return",
    ],
)
def test_plot_rejects_invalid_comparison(case: str, tmp_path: Path) -> None:
    comparison: object = _comparison().copy()
    if case == "not_dataframe":
        comparison = []
    elif case == "columns":
        comparison = _comparison().drop(columns=["spy_buy_hold"])
    elif case == "index_type":
        frame = _comparison().copy()
        frame.index = pd.Index(range(len(frame)), name="Date")
        comparison = frame
    elif case == "index_name":
        frame = _comparison().copy()
        frame.index = frame.index.rename("date")
        comparison = frame
    elif case == "empty":
        comparison = _comparison().iloc[0:0]
    elif case == "duplicate":
        frame = _comparison().copy()
        frame.index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"], name="Date")
        comparison = frame
    elif case == "nonnumeric":
        frame = _comparison().copy()
        frame["regime_rotation"] = "bad"
        comparison = frame
    elif case == "nonfinite":
        frame = _comparison().copy()
        frame.iloc[0, 0] = np.inf
        comparison = frame
    else:
        frame = _comparison().copy()
        frame.iloc[0, 0] = -1.0
        comparison = frame

    error = TypeError if case == "not_dataframe" else ValueError
    with pytest.raises(error):
        plot_cumulative_performance(comparison, tmp_path / "figure.png")  # type: ignore[arg-type]


def test_plot_requires_path_object() -> None:
    with pytest.raises(TypeError):
        plot_cumulative_performance(_comparison(), "figure.png")  # type: ignore[arg-type]
