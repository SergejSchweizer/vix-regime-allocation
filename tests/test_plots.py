from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.plots as plots
from vix_regime_allocation.transform import OUTPUT_COLUMNS

Plotter = Callable[[pd.DataFrame, Path], None]


def _step1_data() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"], name="Date")
    return pd.DataFrame(
        {
            "TLT": [100.0, 101.0, 102.0],
            "GLD": [50.0, 49.0, 51.0],
            "SPY": [200.0, 202.0, 201.0],
            "VIX": [15.0, 16.0, 14.0],
            "TLT_log_return": [0.01, 0.02, -0.01],
            "GLD_log_return": [-0.02, 0.03, 0.01],
            "SPY_log_return": [0.015, -0.005, 0.02],
            "VIX_change": [1.0, 1.0, -2.0],
        },
        index=index,
    ).loc[:, list(OUTPUT_COLUMNS)]


def test_etf_plot_contains_exact_return_series_and_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "etf.png"
    original_close = plots.plt.close
    closed: list[plt.Figure] = []
    monkeypatch.setattr(plots.plt, "close", closed.append)
    plots.plot_etf_log_returns(_step1_data(), output)
    assert output.exists() and output.stat().st_size > 0
    fig = closed[0]
    ax = fig.axes[0]
    data_lines = [line for line in ax.lines if line.get_label() in {"TLT", "GLD", "SPY"}]
    assert [line.get_label() for line in data_lines] == ["TLT", "GLD", "SPY"]
    for line, asset in zip(data_lines, ("TLT", "GLD", "SPY"), strict=True):
        np.testing.assert_allclose(line.get_ydata(), _step1_data()[f"{asset}_log_return"])
    assert ax.get_title() == "ETF Daily Log Returns"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Daily log return"
    assert ax.get_legend() is not None
    original_close(fig)


def test_vix_plot_contains_exact_change_series_and_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "vix.png"
    original_close = plots.plt.close
    closed: list[plt.Figure] = []
    monkeypatch.setattr(plots.plt, "close", closed.append)
    plots.plot_vix_change(_step1_data(), output)
    fig = closed[0]
    ax = fig.axes[0]
    data_line = next(line for line in ax.lines if line.get_label() == "Daily VIX change")
    np.testing.assert_allclose(data_line.get_ydata(), _step1_data()["VIX_change"])
    assert ax.get_title() == "Daily Change in VIX"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Change in VIX (index points)"
    original_close(fig)


@pytest.mark.parametrize("plotter", [plots.plot_etf_log_returns, plots.plot_vix_change])
def test_plotters_close_figures_without_leaks(tmp_path: Path, plotter: Plotter) -> None:
    before = set(plt.get_fignums())
    plotter(_step1_data(), tmp_path / f"{plotter.__name__}.png")
    assert set(plt.get_fignums()) == before


def test_plotters_reject_invalid_data_and_output_path(tmp_path: Path) -> None:
    malformed = _step1_data().drop(columns="VIX_change")
    with pytest.raises(ValueError, match="columns must be exactly"):
        plots.plot_vix_change(malformed, tmp_path / "vix.png")
    non_finite = _step1_data()
    non_finite.loc["2020-01-03", "SPY_log_return"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        plots.plot_etf_log_returns(non_finite, tmp_path / "etf.png")
    with pytest.raises(ValueError, match=".png suffix"):
        plots.plot_etf_log_returns(_step1_data(), tmp_path / "etf.pdf")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        plots.plot_etf_log_returns([], tmp_path / "etf.png")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pathlib.Path"):
        plots.plot_etf_log_returns(_step1_data(), "etf.png")  # type: ignore[arg-type]


def test_plotters_reject_invalid_index_contract(tmp_path: Path) -> None:
    non_datetime = _step1_data().copy()
    non_datetime.index = pd.Index([1, 2, 3], name="Date")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        plots.plot_etf_log_returns(non_datetime, tmp_path / "etf.png")
    unnamed = _step1_data().rename_axis(None)
    with pytest.raises(ValueError, match="named 'Date'"):
        plots.plot_etf_log_returns(unnamed, tmp_path / "etf.png")
    timezone_aware = _step1_data().tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        plots.plot_etf_log_returns(timezone_aware, tmp_path / "etf.png")
    duplicated = _step1_data().copy()
    duplicated.index = pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-06"], name="Date")
    with pytest.raises(ValueError, match="duplicate"):
        plots.plot_etf_log_returns(duplicated, tmp_path / "etf.png")
    unsorted = _step1_data().iloc[[1, 0, 2]]
    with pytest.raises(ValueError, match="sorted"):
        plots.plot_etf_log_returns(unsorted, tmp_path / "etf.png")


def test_plotters_reject_empty_and_non_numeric_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        plots.plot_vix_change(_step1_data().iloc[0:0], tmp_path / "vix.png")
    non_numeric = _step1_data().copy()
    non_numeric["TLT_log_return"] = non_numeric["TLT_log_return"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        plots.plot_etf_log_returns(non_numeric, tmp_path / "etf.png")
