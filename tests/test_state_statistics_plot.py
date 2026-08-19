from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.state_statistics_plot import BASIS_POINTS, plot_state_asset_statistics


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


def test_mean_and_standard_deviation_are_separate_basis_point_panels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[tuple[str, np.ndarray]] = []
    original_bar = plt.Axes.bar

    def recording_bar(self: plt.Axes, *args: object, **kwargs: object) -> object:
        recorded.append((str(kwargs["label"]), np.asarray(args[1], dtype=float)))
        return original_bar(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "bar", recording_bar)
    plot_state_asset_statistics(_statistics(), tmp_path / "plot.png")

    assert [label for label, _ in recorded] == ["TLT", "TLT", "GLD", "GLD", "SPY", "SPY"]
    np.testing.assert_allclose(recorded[0][1], np.array([0.001, 0.003]) * BASIS_POINTS)
    np.testing.assert_allclose(recorded[1][1], np.array([0.01, 0.04]) * BASIS_POINTS)
    np.testing.assert_allclose(recorded[4][1], np.array([-0.001, 0.004]) * BASIS_POINTS)
    np.testing.assert_allclose(recorded[5][1], np.array([0.03, 0.06]) * BASIS_POINTS)


def test_figure_labels_explain_mean_and_dispersion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    original_savefig = plt.Figure.savefig

    def recording_savefig(self: plt.Figure, *args: object, **kwargs: object) -> None:
        mean_axis, std_axis = self.axes
        observed["title"] = mean_axis.get_title()
        observed["mean_ylabel"] = mean_axis.get_ylabel()
        observed["std_ylabel"] = std_axis.get_ylabel()
        observed["xlabel"] = std_axis.get_xlabel()
        observed["mean_zero"] = any(
            np.allclose(line.get_ydata(), [0.0, 0.0]) for line in mean_axis.get_lines()
        )
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", recording_savefig)
    plot_state_asset_statistics(_statistics(), tmp_path / "plot.png")
    assert observed["title"] == "State-conditional ETF return means and dispersion"
    assert observed["mean_ylabel"] == "Mean daily log return (bp)"
    assert observed["std_ylabel"] == "Sample daily standard deviation (bp)"
    assert observed["xlabel"] == "Preferred-model state"
    assert observed["mean_zero"] is True


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
    assert callable(mutator)
    with pytest.raises((TypeError, ValueError)):
        plot_state_asset_statistics(mutator(_statistics()), tmp_path / "plot.png")  # type: ignore[operator]


def test_non_dataframe_fails(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        plot_state_asset_statistics("bad", tmp_path / "plot.png")  # type: ignore[arg-type]
