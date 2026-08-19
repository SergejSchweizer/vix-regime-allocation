from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.markov_plots import plot_markov_vix_states


def _inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.date_range("2020-01-01", periods=6, name="Date")
    vix = pd.Series([12.0, 15.0, 18.0, 14.0, 21.0, 17.0], index=index, name="VIX")
    states_2 = pd.Series([0, 0, 1, 0, 1, 1], index=index, name="state", dtype="int64")
    states_3 = pd.Series([0, 1, 2, 0, 2, 1], index=index, name="state", dtype="int64")
    return vix, states_2, states_3


def test_plot_writes_nonempty_file_and_closes_figure(tmp_path: Path) -> None:
    vix, states_2, states_3 = _inputs()
    output = tmp_path / "nested" / "markov_states.png"
    before = set(plt.get_fignums())
    plot_markov_vix_states(vix, states_2, states_3, output)
    assert output.exists() and output.stat().st_size > 0
    assert set(plt.get_fignums()) == before


def test_each_delta_vix_state_is_plotted_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vix, states_2, states_3 = _inputs()
    labels: list[str] = []
    original_scatter = plt.Axes.scatter

    def recording_scatter(self: plt.Axes, *args: object, **kwargs: object) -> object:
        labels.append(str(kwargs["label"]))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", recording_scatter)
    plot_markov_vix_states(vix, states_2, states_3, tmp_path / "plot.png")
    assert labels == [
        "ΔVIX state 0",
        "ΔVIX state 1",
        "ΔVIX state 0",
        "ΔVIX state 1",
        "ΔVIX state 2",
    ]


def test_index_and_data_validation_failures(tmp_path: Path) -> None:
    vix, states_2, states_3 = _inputs()
    shifted = states_2.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="exactly match"):
        plot_markov_vix_states(vix, shifted, states_3, tmp_path / "x.png")
    with pytest.raises(ValueError, match="named"):
        plot_markov_vix_states(vix, states_2.rename("wrong"), states_3, tmp_path / "x.png")
    bad_labels = states_3.copy()
    bad_labels.iloc[0] = 3
    with pytest.raises(ValueError, match="0..2"):
        plot_markov_vix_states(vix, states_2, bad_labels, tmp_path / "x.png")
    with pytest.raises(ValueError, match="integer"):
        plot_markov_vix_states(vix, states_2.astype(float), states_3, tmp_path / "x.png")
    nonfinite = vix.copy()
    nonfinite.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plot_markov_vix_states(nonfinite, states_2, states_3, tmp_path / "x.png")
