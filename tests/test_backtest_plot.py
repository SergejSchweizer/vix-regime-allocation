from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.backtest_plot as module
from vix_regime_allocation.backtest_plot import (
    PLOT_COLUMNS,
    plot_cumulative_performance,
)


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


def _instrument_returns(comparison: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _comparison() if comparison is None else comparison
    return pd.DataFrame(
        {
            "TLT": [0.005, -0.002, 0.004],
            "GLD": [0.002, 0.006, -0.001],
            "SPY": frame["spy_buy_hold"].to_numpy(dtype=float),
        },
        index=frame.index,
    )


def test_plot_delegates_once_per_series_and_drawdown_includes_initial_wealth(
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
    plot_cumulative_performance(comparison, output, _instrument_returns(comparison))

    assert calls == list(PLOT_COLUMNS)
    assert output.exists() and output.stat().st_size > 0
    assert set(plt.get_fignums()) == before

    spy_wealth = (1.0 + comparison["spy_buy_hold"]).cumprod()
    expected_first_drawdown = spy_wealth.iloc[0] / 1.0 - 1.0
    assert expected_first_drawdown == pytest.approx(-0.01)


def test_plot_auto_loads_all_instruments_from_step1_artifact(tmp_path: Path) -> None:
    comparison = _comparison()
    data_path = tmp_path / "data/processed/step1_data.csv"
    data_path.parent.mkdir(parents=True)
    step1 = pd.DataFrame(
        {
            "Date": comparison.index,
            "TLT_log_return": np.log1p([0.005, -0.002, 0.004]),
            "GLD_log_return": np.log1p([0.002, 0.006, -0.001]),
            "SPY_log_return": np.log1p(comparison["spy_buy_hold"].to_numpy(dtype=float)),
        }
    )
    step1.to_csv(data_path, index=False)

    output = tmp_path / "reports/figures/figure.png"
    plot_cumulative_performance(comparison, output)

    assert output.exists() and output.stat().st_size > 0


def test_saved_figure_has_all_instruments_cumulative_drawdown_and_terminal_panels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    original_savefig = plt.Figure.savefig

    def recording_savefig(self: plt.Figure, *args: object, **kwargs: object) -> None:
        cumulative_axis, drawdown_axis, terminal_axis = self.axes
        observed["title"] = cumulative_axis.get_title()
        observed["cum_ylabel"] = cumulative_axis.get_ylabel()
        observed["dd_title"] = drawdown_axis.get_title()
        observed["dd_ylabel"] = drawdown_axis.get_ylabel()
        observed["xlabel"] = drawdown_axis.get_xlabel()
        observed["terminal_title"] = terminal_axis.get_title()
        observed["terminal_xlabel"] = terminal_axis.get_xlabel()
        observed["cum_lines"] = [
            line.get_label()
            for line in cumulative_axis.lines
            if not str(line.get_label()).startswith("_")
        ]
        observed["dd_lines"] = [
            line.get_label()
            for line in drawdown_axis.lines
            if not str(line.get_label()).startswith("_")
        ]
        observed["terminal_widths"] = [patch.get_width() for patch in terminal_axis.patches]
        observed["endpoint_labels"] = [text.get_text() for text in cumulative_axis.texts]
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", recording_savefig)
    comparison = _comparison()
    instruments = _instrument_returns(comparison)
    plot_cumulative_performance(comparison, tmp_path / "figure.png", instruments)

    assert observed["title"] == "Step 5 Cumulative Performance Comparison — All Instruments"
    assert observed["cum_ylabel"] == "Cumulative return"
    assert observed["dd_title"] == "Drawdown history"
    assert observed["dd_ylabel"] == "Drawdown"
    assert observed["xlabel"] == "Date"
    assert observed["terminal_title"] == "Terminal cumulative return"
    assert observed["terminal_xlabel"] == "Cumulative return"
    assert observed["cum_lines"] == [
        "Regime rotation",
        "Equal weight (monthly reset)",
        "TLT buy and hold",
        "GLD buy and hold",
        "SPY buy and hold",
    ]
    assert observed["dd_lines"] == observed["cum_lines"]

    plot_returns = pd.concat(
        [
            comparison[["regime_rotation", "equal_weight_monthly"]],
            instruments,
        ],
        axis=1,
    ).loc[:, list(PLOT_COLUMNS)]
    expected_terminal = [
        float((1.0 + plot_returns[column]).prod() - 1.0) for column in PLOT_COLUMNS
    ]
    np.testing.assert_allclose(observed["terminal_widths"], expected_terminal)
    assert len(observed["endpoint_labels"]) == len(PLOT_COLUMNS)
    assert all(str(label).endswith("%") for label in observed["endpoint_labels"])


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
        frame.index = pd.DatetimeIndex(
            ["2026-01-02", "2026-01-02", "2026-01-06"],
            name="Date",
        )
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
        plot_cumulative_performance(  # type: ignore[arg-type]
            comparison,
            tmp_path / "figure.png",
            _instrument_returns(),
        )


@pytest.mark.parametrize(
    "case",
    ["not_dataframe", "columns", "index", "nonnumeric", "nonfinite", "invalid_return", "spy"],
)
def test_plot_rejects_invalid_instrument_returns(case: str, tmp_path: Path) -> None:
    comparison = _comparison()
    instruments: object = _instrument_returns(comparison).copy()
    if case == "not_dataframe":
        instruments = []
    elif case == "columns":
        instruments = _instrument_returns(comparison).drop(columns=["GLD"])
    elif case == "index":
        frame = _instrument_returns(comparison).copy()
        frame.index = frame.index + pd.Timedelta(days=1)
        instruments = frame
    elif case == "nonnumeric":
        frame = _instrument_returns(comparison).copy()
        frame["TLT"] = "bad"
        instruments = frame
    elif case == "nonfinite":
        frame = _instrument_returns(comparison).copy()
        frame.iloc[0, 0] = np.inf
        instruments = frame
    elif case == "invalid_return":
        frame = _instrument_returns(comparison).copy()
        frame.iloc[0, 0] = -1.0
        instruments = frame
    else:
        frame = _instrument_returns(comparison).copy()
        frame["SPY"] = frame["SPY"] + 0.001
        instruments = frame

    error = TypeError if case == "not_dataframe" else ValueError
    with pytest.raises(error):
        plot_cumulative_performance(  # type: ignore[arg-type]
            comparison,
            tmp_path / "figure.png",
            instruments,
        )


def test_plot_requires_path_object() -> None:
    with pytest.raises(TypeError):
        plot_cumulative_performance(  # type: ignore[arg-type]
            _comparison(),
            "figure.png",
            _instrument_returns(),
        )
