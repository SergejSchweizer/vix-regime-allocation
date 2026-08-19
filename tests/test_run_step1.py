from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.run_step1 as runner
from vix_regime_allocation.transform import OUTPUT_COLUMNS


def _raw_prices() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"], name="Date")
    return pd.DataFrame(
        {
            "TLT": [100.0, 101.0, 102.0],
            "GLD": [50.0, 51.0, 52.0],
            "SPY": [200.0, 201.0, 202.0],
            "VIX": [15.0, 16.0, 14.0],
        },
        index=index,
    )


def _prepared_data() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="Date")
    return pd.DataFrame(
        {
            "TLT": [101.0, 102.0],
            "GLD": [51.0, 52.0],
            "SPY": [201.0, 202.0],
            "VIX": [16.0, 14.0],
            "TLT_log_return": [0.009950330853168092, 0.00985229644301164],
            "GLD_log_return": [0.01980262729617973, 0.019418085857101516],
            "SPY_log_return": [0.004987541511038968, 0.004962789342129097],
            "VIX_change": [1.0, -2.0],
        },
        index=index,
    ).loc[:, list(OUTPUT_COLUMNS)]


def test_run_step1_delegates_once_and_writes_canonical_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _raw_prices()
    prepared = _prepared_data()
    calls: list[tuple[str, object]] = []

    def fake_download() -> pd.DataFrame:
        calls.append(("download", None))
        return raw

    def fake_prepare(prices: pd.DataFrame) -> pd.DataFrame:
        calls.append(("prepare", prices))
        assert prices is raw
        return prepared

    def fake_etf_plot(data: pd.DataFrame, output_path: Path) -> None:
        calls.append(("etf_plot", output_path))
        assert data is prepared
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"etf")

    def fake_vix_plot(data: pd.DataFrame, output_path: Path) -> None:
        calls.append(("vix_plot", output_path))
        assert data is prepared
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"vix")

    monkeypatch.setattr(runner, "download_adjusted_close", fake_download)
    monkeypatch.setattr(runner, "prepare_step1_data", fake_prepare)
    monkeypatch.setattr(runner, "plot_etf_log_returns", fake_etf_plot)
    monkeypatch.setattr(runner, "plot_vix_change", fake_vix_plot)

    result = runner.run_step1(tmp_path)

    assert result is prepared
    assert [name for name, _ in calls] == ["download", "prepare", "etf_plot", "vix_plot"]
    assert calls[2][1] == tmp_path / runner.ETF_FIGURE_PATH
    assert calls[3][1] == tmp_path / runner.VIX_FIGURE_PATH

    csv_path = tmp_path / runner.STEP1_DATA_PATH
    assert csv_path.exists()
    reloaded = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    reloaded.index.name = "Date"
    pd.testing.assert_frame_equal(reloaded, prepared, check_freq=False)
    assert (tmp_path / runner.ETF_FIGURE_PATH).read_bytes() == b"etf"
    assert (tmp_path / runner.VIX_FIGURE_PATH).read_bytes() == b"vix"

    stdout = capsys.readouterr().out
    assert stdout == "Start: 2020-01-02\nEnd: 2020-01-03\nCount: 2\n"


def test_run_step1_rejects_non_path_output_root() -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        runner.run_step1("output")  # type: ignore[arg-type]
