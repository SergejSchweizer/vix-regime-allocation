from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics


def _statistics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": [0, 0, 0, 1, 1, 1],
            "asset": ["TLT", "GLD", "SPY", "TLT", "GLD", "SPY"],
            "mean_log_return": [0.001, 0.002, -0.001, 0.003, -0.002, 0.004],
            "std_log_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "observations": [10, 10, 10, 12, 12, 12],
        }
    )


def test_plot_writes_nonempty_file_and_closes_figure(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "state_statistics.png"
    before = set(plt.get_fignums())

    plot_state_asset_statistics(_statistics(), output)

    assert output.exists() and output.stat().st_size > 0
    assert set(plt.get_fignums()) == before


def test_bars_use_asset_means_and_standard_deviation_error_bars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[tuple[str, np.ndarray, np.ndarray]] = []
    original_bar = plt.Axes.bar

    def recording_bar(self: plt.Axes, *args: object, **kwargs: object) -> object:
        recorded.append(
            (
                str(kwargs["label"]),
                np.asarray(args[1], dtype=float),
                np.asarray(kwargs["yerr"], dtype=float),
            )
        )
        return original_bar(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "bar", recording_bar)
    plot_state_asset_statistics(_statistics(), tmp_path / "plot.png")

    assert [entry[0] for entry in recorded] == ["TLT", "GLD", "SPY"]
    np.testing.assert_allclose(recorded[0][1], [0.001, 0.003])
    np.testing.assert_allclose(recorded[0][2], [0.01, 0.04])
    np.testing.assert_allclose(recorded[1][1], [0.002, -0.002])
    np.testing.assert_allclose(recorded[1][2], [0.02, 0.05])
    np.testing.assert_allclose(recorded[2][1], [-0.001, 0.004])
    np.testing.assert_allclose(recorded[2][2], [0.03, 0.06])


def test_figure_has_zero_line_labels_title_and_legend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    original_savefig = plt.Figure.savefig

    def recording_savefig(self: plt.Figure, *args: object, **kwargs: object) -> None:
        axis = self.axes[0]
        observed["xlabel"] = axis.get_xlabel()
        observed["ylabel"] = axis.get_ylabel()
        observed["title"] = axis.get_title()
        observed["legend"] = axis.get_legend() is not None
        observed["zero_line"] = any(
            np.allclose(line.get_ydata(), [0.0, 0.0]) for line in axis.get_lines()
        )
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", recording_savefig)
    plot_state_asset_statistics(_statistics(), tmp_path / "plot.png")

    assert observed["xlabel"] == "Preferred-model state"
    assert observed["ylabel"] == "Daily ETF log return"
    assert "State-conditional ETF mean daily log returns" in str(observed["title"])
    assert observed["legend"] is True
    assert observed["zero_line"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns="observations"),
        lambda frame: frame.assign(state=frame["state"].astype(float)),
        lambda frame: frame.assign(observations=frame["observations"].astype(float)),
        lambda frame: frame.assign(mean_log_return=np.nan),
        lambda frame: frame.assign(std_log_return=-0.01),
        lambda frame: frame.assign(observations=0),
        lambda frame: frame.iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True),
        lambda frame: frame.assign(asset=["TLT", "GLD", "QQQ", "TLT", "GLD", "SPY"]),
        lambda frame: frame.assign(state=[0, 0, 0, 2, 2, 2]),
    ],
)
def test_invalid_statistics_fail(mutator: object, tmp_path: Path) -> None:
    transform = mutator
    assert callable(transform)
    with pytest.raises((TypeError, ValueError)):
        plot_state_asset_statistics(transform(_statistics()), tmp_path / "plot.png")  # type: ignore[operator]


def test_non_dataframe_fails(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        plot_state_asset_statistics("bad", tmp_path / "plot.png")  # type: ignore[arg-type]
