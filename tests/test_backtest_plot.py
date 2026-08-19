from __future__ import annotations

from pathlib import Path

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


class _FakeAxis:
    def __init__(self) -> None:
        self.plots: list[tuple[object, np.ndarray, str]] = []
        self.zero_line: tuple[float, float] | None = None
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.legend_called = False
        self.grid_called = False

    def plot(self, x: object, y: pd.Series, *, label: str) -> None:
        self.plots.append((x, y.to_numpy(dtype=float), label))

    def axhline(self, y: float, *, linewidth: float) -> None:
        self.zero_line = (y, linewidth)

    def set_title(self, value: str) -> None:
        self.title = value

    def set_xlabel(self, value: str) -> None:
        self.xlabel = value

    def set_ylabel(self, value: str) -> None:
        self.ylabel = value

    def legend(self) -> None:
        self.legend_called = True

    def grid(self, enabled: bool, *, alpha: float) -> None:
        self.grid_called = enabled and alpha == 0.25


class _FakeFigure:
    def __init__(self) -> None:
        self.tight_called = False
        self.saved: tuple[Path, int, str] | None = None

    def tight_layout(self) -> None:
        self.tight_called = True

    def savefig(self, path: Path, *, dpi: int, bbox_inches: str) -> None:
        self.saved = (path, dpi, bbox_inches)


def test_plot_delegates_to_shared_wealth_and_renders_exact_three_curves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comparison = _comparison()
    axis = _FakeAxis()
    figure = _FakeFigure()
    calls: list[str] = []
    closed: list[object] = []

    def fake_wealth(series: pd.Series) -> pd.Series:
        calls.append(str(series.name))
        return (1.0 + series).cumprod().rename("wealth")

    monkeypatch.setattr(module, "cumulative_wealth", fake_wealth)
    monkeypatch.setattr(module.plt, "subplots", lambda **_: (figure, axis))
    monkeypatch.setattr(module.plt, "close", lambda fig: closed.append(fig))

    output = tmp_path / "nested" / "figure.png"
    plot_cumulative_performance(comparison, output)

    assert calls == list(COMPARISON_COLUMNS)
    assert len(axis.plots) == 3
    for column, (_, y_values, _) in zip(COMPARISON_COLUMNS, axis.plots, strict=True):
        expected = (1.0 + comparison[column]).cumprod() - 1.0
        np.testing.assert_allclose(y_values, expected.to_numpy())
    assert [entry[2] for entry in axis.plots] == [
        "Regime rotation",
        "Equal weight (monthly reset)",
        "SPY buy and hold",
    ]
    assert axis.zero_line == (0.0, 0.8)
    assert axis.title == "Cumulative Performance Comparison"
    assert axis.xlabel == "Date"
    assert axis.ylabel == "Cumulative return"
    assert axis.legend_called
    assert axis.grid_called
    assert figure.tight_called
    assert figure.saved == (output, 160, "tight")
    assert output.parent.exists()
    assert closed == [figure]


def test_plot_closes_figure_when_save_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    comparison = _comparison()
    axis = _FakeAxis()
    figure = _FakeFigure()
    closed: list[object] = []

    def fail_save(*_: object, **__: object) -> None:
        raise OSError("disk full")

    figure.savefig = fail_save  # type: ignore[method-assign]
    monkeypatch.setattr(module.plt, "subplots", lambda **_: (figure, axis))
    monkeypatch.setattr(module.plt, "close", lambda fig: closed.append(fig))

    with pytest.raises(OSError, match="disk full"):
        plot_cumulative_performance(comparison, tmp_path / "figure.png")
    assert closed == [figure]


@pytest.mark.parametrize(
    "case", ["not_dataframe", "columns", "index_type", "index_name", "empty", "duplicate", "nonnumeric", "nonfinite", "invalid_return"]
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
