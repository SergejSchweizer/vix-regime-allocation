from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vix_regime_allocation.predictive.artifacts as artifacts_module
import vix_regime_allocation.predictive.hmm_filter as hmm_filter_module
from vix_regime_allocation.predictive.artifacts import PredictiveAnalysis, compute_predictive_analysis
from vix_regime_allocation.predictive.dominance import compare_against_assets
from vix_regime_allocation.predictive.hmm_filter import HMMFilterModel, filtered_probabilities
from vix_regime_allocation.predictive.markov_forecast import (
    MarkovForecastModel,
    fit_markov_forecaster,
    forecast_next_regime as markov_forecast_next_regime,
)
from vix_regime_allocation.predictive.plots import comparison_return_frame, probability_columns
from vix_regime_allocation.predictive.policy import (
    apply_transaction_cost,
    choose_asset,
    one_hot_weights,
    turnover,
)
from vix_regime_allocation.predictive.returns import asset_simple_returns, buy_and_hold_returns
from vix_regime_allocation.predictive.split import split_periods
from vix_regime_allocation.predictive.state_returns import (
    expected_asset_returns,
    hard_state_asset_means,
    soft_state_asset_means,
)


def _data() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=6, name="Date")
    return pd.DataFrame(
        {
            "TLT_log_return": np.linspace(-0.01, 0.01, 6),
            "GLD_log_return": np.linspace(0.01, -0.01, 6),
            "SPY_log_return": np.linspace(-0.02, 0.02, 6),
        },
        index=index,
    )


def test_return_and_policy_error_contracts() -> None:
    data = _data()
    with pytest.raises(TypeError):
        asset_simple_returns(pd.Series([1.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        asset_simple_returns(pd.DataFrame(index=data.index))
    bad = data.copy()
    bad.index = range(len(bad))
    with pytest.raises(ValueError):
        asset_simple_returns(bad)
    bad = data.copy()
    bad["TLT_log_return"] = ["x"] * len(bad)
    with pytest.raises(ValueError):
        asset_simple_returns(bad)
    bad = data.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        asset_simple_returns(bad)

    simple = asset_simple_returns(data)
    with pytest.raises(TypeError):
        buy_and_hold_returns(pd.Series([1.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        buy_and_hold_returns(simple[["SPY", "GLD", "TLT"]])

    expected = pd.Series([0.01, 0.02, 0.03], index=["TLT", "GLD", "SPY"])
    with pytest.raises(TypeError):
        choose_asset(pd.DataFrame(), None, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        choose_asset(expected.reindex(["SPY", "GLD", "TLT"]), None, 0.0)
    with pytest.raises(ValueError):
        choose_asset(expected, None, -1.0)
    with pytest.raises(ValueError):
        choose_asset(expected, "QQQ", 0.0)
    assert choose_asset(expected, "SPY", 10.0) == "SPY"
    assert choose_asset(expected, "GLD", 200.0) == "GLD"
    with pytest.raises(ValueError):
        one_hot_weights("QQQ")
    for previous, new in [
        (np.zeros(2), np.ones(2)),
        (np.array([np.nan, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.zeros(3), np.array([1.0, -0.1, 0.1])),
        (np.zeros(3), np.array([0.5, 0.3, 0.1])),
        (np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    ]:
        with pytest.raises(ValueError):
            turnover(previous, new)
    for gross, turn, cost in [(-1.0, 0.0, 5.0), (0.0, -1.0, 5.0), (0.0, 0.0, -1.0)]:
        with pytest.raises(ValueError):
            apply_transaction_cost(gross, turn, cost)


def test_state_return_error_contracts() -> None:
    simple = asset_simple_returns(_data())
    states = pd.Series([0, 0, 0, 1, 1, 1], index=simple.index)
    with pytest.raises(TypeError):
        hard_state_asset_means(pd.Series([1.0]), states, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hard_state_asset_means(simple[["SPY", "GLD", "TLT"]], states, 2)
    with pytest.raises(ValueError):
        hard_state_asset_means(simple, states.reset_index(drop=True), 2)
    with pytest.raises(ValueError):
        hard_state_asset_means(simple, pd.Series([0] * 6, index=simple.index), 2)

    probabilities = pd.DataFrame(
        {"state_0": [0.5] * 6, "state_1": [0.5] * 6}, index=simple.index
    )
    with pytest.raises(TypeError):
        soft_state_asset_means(simple, pd.Series([1.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        soft_state_asset_means(simple, probabilities.reset_index(drop=True))
    with pytest.raises(ValueError):
        soft_state_asset_means(simple, probabilities.rename(columns={"state_1": "bad"}))
    invalid = probabilities.copy()
    invalid.iloc[0] = [0.9, 0.9]
    with pytest.raises(ValueError):
        soft_state_asset_means(simple, invalid)

    means = hard_state_asset_means(simple, states, 2)
    for probabilities_value, means_value in [
        (np.array([1.0]), means),
        (np.array([0.9, 0.9]), means),
        (np.array([0.5, 0.5]), means[["SPY", "GLD", "TLT"]]),
    ]:
        with pytest.raises(ValueError):
            expected_asset_returns(probabilities_value, means_value)


def test_forecaster_filter_and_comparison_error_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=6, name="Date")
    series = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], index=index)
    for value, states in [(series, 4), (pd.Series([1.0] * 6, index=index), 2)]:
        with pytest.raises(ValueError):
            fit_markov_forecaster(value, states)
    with pytest.raises(TypeError):
        fit_markov_forecaster(pd.DataFrame(), 2)  # type: ignore[arg-type]
    bad = series.copy()
    bad.index = range(6)
    with pytest.raises(ValueError):
        fit_markov_forecaster(bad, 2)

    model = fit_markov_forecaster(series, 2)
    with pytest.raises(ValueError):
        markov_forecast_next_regime(model, np.nan)
    invalid_model = MarkovForecastModel(
        2,
        model.thresholds,
        np.array([[0.2, 0.2], [0.5, 0.5]]),
        model.training_states,
    )
    with pytest.raises(ValueError):
        markov_forecast_next_regime(invalid_model, -1.0)

    hmm = HMMFilterModel(
        2,
        np.array([0.5, 0.5]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
        np.array([-1.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    with pytest.raises(TypeError):
        filtered_probabilities(hmm, pd.DataFrame())  # type: ignore[arg-type]
    bad = series.iloc[:3].copy()
    bad.index = range(3)
    with pytest.raises(ValueError):
        filtered_probabilities(hmm, bad)
    with pytest.raises(ValueError):
        filtered_probabilities(hmm, pd.Series([np.nan], index=index[:1]))
    fake = SimpleNamespace(
        start_probabilities=(0.5, 0.5),
        transition_matrix=pd.DataFrame([[0.8, 0.2], [0.2, 0.8]]),
        means=pd.Series([-1.0, 1.0]),
        variances=pd.Series([1.0, 1.0]),
    )
    monkeypatch.setattr(hmm_filter_module, "fit_gaussian_hmm", lambda values, k: fake)
    with pytest.raises(ValueError):
        hmm_filter_module.fit_hmm_filter(series, 4)
    assert hmm_filter_module.fit_hmm_filter(series, 2).n_states == 2

    with pytest.raises(TypeError):
        split_periods(pd.Series([1.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        split_periods(pd.DataFrame(index=range(5)))
    with pytest.raises(ValueError):
        split_periods(pd.DataFrame(index=pd.date_range("2020-01-01", periods=3)))

    strategy = pd.Series([0.01] * 4, index=pd.date_range("2021-01-01", periods=4))
    assets = pd.DataFrame(
        {"TLT": [0.0] * 4, "GLD": [0.0] * 4, "SPY": [0.0] * 4}, index=strategy.index
    )
    with pytest.raises(TypeError):
        compare_against_assets(pd.DataFrame(), assets)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compare_against_assets(strategy, assets[["SPY", "GLD", "TLT"]])
    with pytest.raises(ValueError):
        compare_against_assets(strategy, assets.reset_index(drop=True))
    with pytest.raises(ValueError, match="contiguous"):
        probability_columns(pd.DataFrame({"p_state_0": [1.0]}))
    with pytest.raises(ValueError, match="contiguous"):
        probability_columns(pd.DataFrame({"p_state_0": [0.5], "p_state_2": [0.5]}))
    with pytest.raises(ValueError):
        comparison_return_frame(pd.DataFrame(), pd.DataFrame())


def test_predictive_analysis_coordinator_uses_frozen_validation_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2014-12-30",
                "2014-12-31",
                "2015-01-02",
                "2015-01-05",
                "2015-01-06",
                "2020-12-31",
                "2021-01-04",
                "2021-01-05",
                "2021-01-06",
            ]
        ),
        name="Date",
    )
    data = pd.DataFrame(index=index)
    periods = SimpleNamespace(initial_history=index[:2], validation=index[2:6], test=index[6:])
    monkeypatch.setattr(artifacts_module, "split_periods", lambda frame: periods)

    real_dispatch = artifacts_module._signals_for_family

    def fake_signals(
        frame: pd.DataFrame, decisions: pd.DatetimeIndex, family: str, n_states: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "decision_date": decisions,
                "return_date": index[index.get_indexer(decisions) + 1],
                "family": family,
                "n_states": n_states,
                "training_end": [index[0]] * len(decisions),
                "p_state_0": [0.5] * len(decisions),
                "p_state_1": [0.5] * len(decisions),
                "expected_TLT": [0.0] * len(decisions),
                "expected_GLD": [0.0] * len(decisions),
                "expected_SPY": [0.0] * len(decisions),
            }
        )

    monkeypatch.setattr(artifacts_module, "_signals_for_family", fake_signals)
    monkeypatch.setattr(
        artifacts_module,
        "asset_simple_returns",
        lambda frame: pd.DataFrame(
            {
                "TLT": [0.0] * len(index),
                "GLD": [0.0] * len(index),
                "SPY": [0.0] * len(index),
            },
            index=index,
        ),
    )
    validation = pd.DataFrame(
        {
            "family": ["markov"],
            "n_states": [2],
            "switch_hurdle_bps": [5.0],
            "mean_log_growth": [0.001],
            "mean_turnover": [0.1],
            "selected": [True],
        }
    )
    monkeypatch.setattr(
        artifacts_module,
        "build_validation_summary",
        lambda signals, returns: validation,
    )
    monkeypatch.setattr(
        artifacts_module,
        "selected_configuration",
        lambda summary: ("markov", 2, 5.0),
    )
    daily = pd.DataFrame(
        {
            "decision_date": index[6:8],
            "return_date": index[7:9],
            "family": ["markov", "markov"],
            "n_states": [2, 2],
            "switch_hurdle_bps": [5.0, 5.0],
            "net_return": [0.0, 0.0],
        }
    )
    holdout = SimpleNamespace(
        daily=daily,
        performance=pd.DataFrame({"portfolio": ["selected_predictive_net"]}),
        dominance=pd.DataFrame(
            {
                "benchmark": ["TLT", "GLD", "SPY"],
                "benchmark_cagr": [0.0, 0.0, 0.0],
                "strategy_net_cagr": [0.0, 0.0, 0.0],
                "cagr_difference": [0.0, 0.0, 0.0],
            }
        ),
        cagr_dominance_margin=0.0,
        dominates_all_individual_assets=False,
    )
    monkeypatch.setattr(artifacts_module, "run_final_holdout", lambda *args, **kwargs: holdout)

    analysis = compute_predictive_analysis(data)
    assert isinstance(analysis, PredictiveAnalysis)
    assert analysis.selected_strategy["family"] == "markov"
    assert analysis.selected_strategy["n_states"] == 2
    assert analysis.selected_strategy["switch_hurdle_bps"] == 5.0
    assert len(analysis.selected_test_daily) == 2

    monkeypatch.setattr(artifacts_module, "_signals_for_family", real_dispatch)
    with pytest.raises(ValueError, match="family"):
        artifacts_module._signals_for_family(data, index[2:4], "invalid", 2)
