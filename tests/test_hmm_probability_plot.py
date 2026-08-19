from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vix_regime_allocation.hmm_probability_plot import plot_hmm_smoothed_probabilities


def _probabilities(n_states: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=5, name="Date")
    if n_states == 2:
        values = np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])
    else:
        values = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
                [0.4, 0.4, 0.2],
                [0.2, 0.2, 0.6],
            ]
        )
    return pd.DataFrame(values, index=index, columns=[f"state_{i}" for i in range(n_states)])


def test_probability_plot_writes_file_and_closes(tmp_path: Path) -> None:
    p2, p3 = _probabilities(2), _probabilities(3)
    output = tmp_path / "nested" / "probabilities.png"
    before = set(plt.get_fignums())
    plot_hmm_smoothed_probabilities(p2, p3, output)
    assert output.exists() and output.stat().st_size > 0
    assert set(plt.get_fignums()) == before


def test_every_probability_column_is_plotted_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p2, p3 = _probabilities(2), _probabilities(3)
    labels: list[str] = []
    original_plot = plt.Axes.plot

    def recording_plot(self: plt.Axes, *args: object, **kwargs: object) -> object:
        labels.append(str(kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", recording_plot)
    plot_hmm_smoothed_probabilities(p2, p3, tmp_path / "plot.png")
    assert labels == ["State 0", "State 1", "State 0", "State 1", "State 2"]


def test_probability_validation_rejects_malformed_inputs(tmp_path: Path) -> None:
    p2, p3 = _probabilities(2), _probabilities(3)
    wrong_columns = p2.rename(columns={"state_1": "bad"})
    with pytest.raises(ValueError, match="columns"):
        plot_hmm_smoothed_probabilities(wrong_columns, p3, tmp_path / "x.png")

    nonnormalized = p2.copy()
    nonnormalized.iloc[0] = [0.6, 0.6]
    with pytest.raises(ValueError, match="sum to one"):
        plot_hmm_smoothed_probabilities(nonnormalized, p3, tmp_path / "x.png")

    out_of_range = p3.copy()
    out_of_range.iloc[0] = [1.1, -0.1, 0.0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_hmm_smoothed_probabilities(p2, out_of_range, tmp_path / "x.png")

    nonfinite = p3.copy()
    nonfinite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plot_hmm_smoothed_probabilities(p2, nonfinite, tmp_path / "x.png")

    unsorted = p2.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        plot_hmm_smoothed_probabilities(unsorted, p3, tmp_path / "x.png")
