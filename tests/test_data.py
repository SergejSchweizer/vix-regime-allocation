from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation import data

YAHOO_TICKERS = ["TLT", "GLD", "SPY", "^VIX"]


def _download_frame(
    index: pd.DatetimeIndex,
    *,
    adjusted_values: np.ndarray | None = None,
) -> pd.DataFrame:
    if adjusted_values is None:
        adjusted_values = np.array(
            [
                [90.0, 180.0, 500.0, 15.0],
                [91.0, 181.0, 501.0, 16.0],
                [92.0, 182.0, 502.0, 17.0],
            ]
        )
    open_values = adjusted_values + 1.0
    values = np.concatenate([open_values, adjusted_values], axis=1)
    columns = pd.MultiIndex.from_product([["Open", "Adj Close"], YAHOO_TICKERS])
    return pd.DataFrame(values, index=index, columns=columns)


def _install_download_mock(
    monkeypatch: pytest.MonkeyPatch,
    frame_factory: Callable[[], pd.DataFrame],
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return frame_factory()

    monkeypatch.setattr(data.yf, "download", fake_download)
    return calls


def test_download_uses_exact_contract_and_normalizes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-01", "2024-01-02"], tz="UTC", name="source_date"
    )
    frame = _download_frame(index)
    calls = _install_download_mock(monkeypatch, lambda: frame)

    result = data.download_adjusted_close()

    assert calls == [
        {
            "tickers": YAHOO_TICKERS,
            "period": "max",
            "interval": "1d",
            "auto_adjust": False,
            "back_adjust": False,
            "actions": False,
            "progress": False,
        }
    ]
    assert list(result.columns) == ["TLT", "GLD", "SPY", "VIX"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "Date"
    assert result.index.tz is None
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique

    expected = frame["Adj Close"].rename(columns={"^VIX": "VIX"})
    expected.index = expected.index.tz_localize(None)
    expected.index.name = "Date"
    expected = expected.sort_index()
    pd.testing.assert_frame_equal(result, expected)


def test_missing_values_are_preserved_for_common_sample_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2024-01-01", periods=3, name="Date")
    values = np.array(
        [
            [90.0, 180.0, 500.0, 15.0],
            [91.0, np.nan, 501.0, 16.0],
            [92.0, 182.0, 502.0, 17.0],
        ]
    )
    calls = _install_download_mock(
        monkeypatch, lambda: _download_frame(index, adjusted_values=values)
    )

    result = data.download_adjusted_close()

    assert len(calls) == 1
    assert np.isnan(result.loc[pd.Timestamp("2024-01-02"), "GLD"])


def test_duplicate_dates_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    _install_download_mock(monkeypatch, lambda: _download_frame(index))

    with pytest.raises(ValueError, match="duplicate dates"):
        data.download_adjusted_close()


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf, -np.inf])
def test_invalid_non_missing_prices_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    values = np.array(
        [
            [90.0, 180.0, 500.0, 15.0],
            [91.0, 181.0, 501.0, 16.0],
            [92.0, 182.0, 502.0, 17.0],
        ]
    )
    values[1, 2] = bad_value
    _install_download_mock(monkeypatch, lambda: _download_frame(index, adjusted_values=values))

    with pytest.raises(ValueError, match="finite|strictly positive"):
        data.download_adjusted_close()


def test_missing_adjusted_close_field_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    frame = _download_frame(index).loc[:, ["Open"]]
    _install_download_mock(monkeypatch, lambda: frame)

    with pytest.raises(ValueError, match="Adj Close"):
        data.download_adjusted_close()


def test_missing_required_ticker_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    frame = _download_frame(index).drop(columns=("Adj Close", "^VIX"))
    _install_download_mock(monkeypatch, lambda: frame)

    with pytest.raises(ValueError, match="missing required tickers"):
        data.download_adjusted_close()


def test_flat_columns_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"Adj Close": [1.0]}, index=[pd.Timestamp("2024-01-01")])
    _install_download_mock(monkeypatch, lambda: frame)

    with pytest.raises(ValueError, match="MultiIndex"):
        data.download_adjusted_close()


def test_non_datetime_index_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _download_frame(pd.Index([object(), object(), object()]))  # type: ignore[arg-type]
    _install_download_mock(monkeypatch, lambda: frame)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        data.download_adjusted_close()


def test_non_numeric_adjusted_close_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    frame = _download_frame(index)
    frame[("Adj Close", "GLD")] = ["bad", "bad", "bad"]
    _install_download_mock(monkeypatch, lambda: frame)

    with pytest.raises(ValueError, match="must be numeric"):
        data.download_adjusted_close()


def test_non_dataframe_download_result_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(**kwargs: object) -> object:
        return object()

    monkeypatch.setattr(data.yf, "download", fake_download)

    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        data.download_adjusted_close()
